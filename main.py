import os
import json
import torch
import torchvision
import torch.nn.parallel
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import opts_egtea as opts
import time
import h5py
from tqdm import tqdm
from iou_utils import *
from eval import evaluation_detection
from tensorboardX import SummaryWriter
from dataset import VideoDataSet
from models import MYNET, SuppressNet
from loss_func import cls_loss_func, cls_loss_func_, regress_loss_func
from loss_func import MultiCrossEntropyLoss
from functools import *

def train_one_epoch(opt, model, train_dataset, optimizer, warmup=False):
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=opt['batch_size'], shuffle=True,
                                                num_workers=0, pin_memory=True,drop_last=False)      
    epoch_cost = 0
    epoch_cost_cls = 0
    epoch_cost_reg = 0
    epoch_cost_snip = 0
    
    total_iter = len(train_dataset)//opt['batch_size']
    cls_loss = MultiCrossEntropyLoss(focal=True)
    snip_loss = MultiCrossEntropyLoss(focal=True)
    for n_iter,(input_data,cls_label,reg_label,snip_label) in enumerate(tqdm(train_loader)):

        if warmup:
            for g in optimizer.param_groups:
                g['lr'] = n_iter * (opt['lr']) / total_iter
        
        act_cls, act_reg, snip_cls = model(input_data.cuda())

        
        act_cls.register_hook(partial(cls_loss.collect_grad, cls_label))
        snip_cls.register_hook(partial(snip_loss.collect_grad, snip_label))
        
        cost_reg = 0
        cost_cls = 0

        loss = cls_loss_func_(cls_loss, cls_label,act_cls)
        cost_cls = loss
            
        epoch_cost_cls+= cost_cls.detach().cpu().numpy()    
               
        loss = regress_loss_func(reg_label,act_reg)
        cost_reg = loss  
        epoch_cost_reg += cost_reg.detach().cpu().numpy()   

        loss = cls_loss_func_(snip_loss, snip_label,snip_cls)
        cost_snip = loss

            
        epoch_cost_snip+= cost_snip.detach().cpu().numpy() 
        
        cost= opt['alpha']*cost_cls +opt['beta']*cost_reg + opt['gamma']*cost_snip    
                
        epoch_cost += cost.detach().cpu().numpy() 

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()   
                
    return n_iter, epoch_cost, epoch_cost_cls, epoch_cost_reg, epoch_cost_snip

def eval_one_epoch(opt, model, test_dataset):
    cls_loss, reg_loss, tot_loss, output_cls, output_reg, labels_cls, labels_reg, working_time, total_frames = eval_frame(opt, model,test_dataset)
        
    result_dict = eval_map_nms(opt,test_dataset, output_cls, output_reg, labels_cls, labels_reg)
    output_dict={"version":"VERSION 1.3","results":result_dict,"external_data":{}}
    outfile=open(opt["result_file"].format(opt['exp']),"w")
    json.dump(output_dict,outfile, indent=2)
    outfile.close()
    
    IoUmAP = evaluation_detection(opt, verbose=False)
    IoUmAP_5=sum(IoUmAP[0:])/len(IoUmAP[0:])

    return cls_loss, reg_loss, tot_loss, IoUmAP_5

    
def train(opt): 
    writer = SummaryWriter()
    model = MYNET(opt).cuda()
    
    rest_of_model_params = [param for name, param in model.named_parameters() if "history_unit" not in name]
  
    optimizer = optim.Adam([{'params': model.history_unit.parameters(), 'lr': 1e-6}, {'params': rest_of_model_params}],lr=opt["lr"],weight_decay = opt["weight_decay"])  
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size = opt["lr_step"])
    
    train_dataset = VideoDataSet(opt,subset="train")      
    test_dataset = VideoDataSet(opt,subset=opt['inference_subset'])
    
    warmup=False
    
    for n_epoch in range(opt['epoch']):   
        if n_epoch >=1:
            warmup=False
        
        n_iter, epoch_cost, epoch_cost_cls, epoch_cost_reg, epoch_cost_snip = train_one_epoch(opt, model, train_dataset, optimizer, warmup)
            
        writer.add_scalars('data/cost', {'train': epoch_cost/(n_iter+1)}, n_epoch)
        print("training loss(epoch %d): %.03f, cls - %f, reg - %f, snip - %f, lr - %f"%(n_epoch,
                                                                            epoch_cost/(n_iter+1),
                                                                            epoch_cost_cls/(n_iter+1),
                                                                            epoch_cost_reg/(n_iter+1),
                                                                            epoch_cost_snip/(n_iter+1),
                                                                            optimizer.param_groups[-1]["lr"]) )
        
        scheduler.step()
        model.eval()
        
        cls_loss, reg_loss, tot_loss, IoUmAP_5 = eval_one_epoch(opt, model,test_dataset)
        
        writer.add_scalars('data/mAP', {'test': IoUmAP_5}, n_epoch)
        print("testing loss(epoch %d): %.03f, cls - %f, reg - %f, mAP Avg - %f"%(n_epoch,tot_loss, cls_loss, reg_loss, IoUmAP_5))
                    
        state = {'epoch': n_epoch + 1,
                    'state_dict': model.state_dict()}
        torch.save(state, opt["checkpoint_path"]+"/"+opt["exp"]+"_checkpoint_"+str(n_epoch+1)+".pth.tar" )
        if IoUmAP_5 > model.best_map:
            model.best_map = IoUmAP_5
            torch.save(state, opt["checkpoint_path"]+"/"+opt["exp"]+"_ckp_best.pth.tar" )
            
        model.train()
                
    writer.close()
    return model.best_map

def eval_frame(opt, model, dataset):
    test_loader = torch.utils.data.DataLoader(dataset,
                                                batch_size=opt['batch_size'], shuffle=False,
                                                num_workers=0, pin_memory=True,drop_last=False)
    
    labels_cls={}
    labels_reg={}
    output_cls={}
    output_reg={}                                      
    for video_name in dataset.video_list:
        labels_cls[video_name]=[]
        labels_reg[video_name]=[]
        output_cls[video_name]=[]
        output_reg[video_name]=[]
        
    start_time = time.time()
    total_frames =0  
    epoch_cost = 0
    epoch_cost_cls = 0
    epoch_cost_reg = 0   
    
    for n_iter,(input_data,cls_label,reg_label, _) in enumerate(tqdm(test_loader)):
        act_cls, act_reg, _ = model(input_data.cuda())
        cost_reg = 0
        cost_cls = 0
        
        loss = cls_loss_func(cls_label,act_cls)
        cost_cls = loss
            
        epoch_cost_cls+= cost_cls.detach().cpu().numpy()    
               
        loss = regress_loss_func(reg_label,act_reg)
        cost_reg = loss  
        epoch_cost_reg += cost_reg.detach().cpu().numpy()   
        
        cost= opt['alpha']*cost_cls +opt['beta']*cost_reg    
                
        epoch_cost += cost.detach().cpu().numpy() 
        
        act_cls = torch.softmax(act_cls, dim=-1)
        
        total_frames+=input_data.size(0)
        
        for b in range(0,input_data.size(0)):
            video_name, st, ed, data_idx = dataset.inputs[n_iter*opt['batch_size']+b]
            output_cls[video_name]+=[act_cls[b,:].detach().cpu().numpy()]
            output_reg[video_name]+=[act_reg[b,:].detach().cpu().numpy()]
            labels_cls[video_name]+=[cls_label[b,:].numpy()]
            labels_reg[video_name]+=[reg_label[b,:].numpy()]
        
    end_time = time.time()
    working_time = end_time-start_time
    
    for video_name in dataset.video_list:
        labels_cls[video_name]=np.stack(labels_cls[video_name], axis=0)
        labels_reg[video_name]=np.stack(labels_reg[video_name], axis=0)
        output_cls[video_name]=np.stack(output_cls[video_name], axis=0)
        output_reg[video_name]=np.stack(output_reg[video_name], axis=0)
    
    cls_loss=epoch_cost_cls/n_iter
    reg_loss=epoch_cost_reg/n_iter
    tot_loss=epoch_cost/n_iter
     
    return cls_loss, reg_loss, tot_loss, output_cls, output_reg, labels_cls, labels_reg, working_time, total_frames


def eval_map_nms(opt, dataset, output_cls, output_reg, labels_cls, labels_reg):
    result_dict={}
    proposal_dict=[]
    
    num_class = opt["num_of_class"]
    unit_size = opt['segment_size']
    threshold=opt['threshold']
    anchors=opt['anchors']
                                             
    for video_name in dataset.video_list:
        duration = dataset.video_len[video_name]
        video_time = float(dataset.video_dict[video_name]["duration"])
        frame_to_time = 100.0*video_time / duration
         
        for idx in range(0,duration):
            cls_anc = output_cls[video_name][idx]
            reg_anc = output_reg[video_name][idx]
            
            proposal_anc_dict=[]
            for anc_idx in range(0,len(anchors)):
                cls = np.argwhere(cls_anc[anc_idx][:-1]>opt['threshold']).reshape(-1)
                
                if len(cls) == 0:
                    continue
                    
                ed= idx + anchors[anc_idx] * reg_anc[anc_idx][0]
                length = anchors[anc_idx]* np.exp(reg_anc[anc_idx][1])
                st= ed-length
                
                for cidx in range(0,len(cls)):
                    label=cls[cidx]
                    tmp_dict={}
                    tmp_dict["segment"] = [st*frame_to_time/100.0, ed*frame_to_time/100.0]
                    tmp_dict["score"]= cls_anc[anc_idx][label]*1.0
                    tmp_dict["label"]=dataset.label_name[label]
                    tmp_dict["gentime"]= idx*frame_to_time/100.0
                    proposal_anc_dict.append(tmp_dict)
                
            proposal_dict+=proposal_anc_dict
        
        proposal_dict=non_max_suppression(proposal_dict, overlapThresh=opt['soft_nms'])
                    
        result_dict[video_name]=proposal_dict
        proposal_dict=[]
        
    return result_dict


def eval_map_supnet(opt, dataset, output_cls, output_reg, labels_cls, labels_reg):
    model = SuppressNet(opt).cuda()
    checkpoint = torch.load(opt["checkpoint_path"]+"/ckp_best_suppress.pth.tar")
    base_dict=checkpoint['state_dict']
    model.load_state_dict(base_dict)
    model.eval()
    
    result_dict={}
    proposal_dict=[]
    
    num_class = opt["num_of_class"]
    unit_size = opt['segment_size']
    threshold=opt['threshold']
    anchors=opt['anchors']
                                             
    for video_name in dataset.video_list:
        duration = dataset.video_len[video_name]
        video_time = float(dataset.video_dict[video_name]["duration"])
        frame_to_time = 100.0*video_time / duration
        conf_queue = torch.zeros((unit_size,num_class-1)) 
        
        for idx in range(0,duration):
            cls_anc = output_cls[video_name][idx]
            reg_anc = output_reg[video_name][idx]
            
            proposal_anc_dict=[]
            for anc_idx in range(0,len(anchors)):
                cls = np.argwhere(cls_anc[anc_idx][:-1]>opt['threshold']).reshape(-1)
                
                if len(cls) == 0:
                    continue
                    
                ed= idx + anchors[anc_idx] * reg_anc[anc_idx][0]
                length = anchors[anc_idx]* np.exp(reg_anc[anc_idx][1])
                st= ed-length
                
                for cidx in range(0,len(cls)):
                    label=cls[cidx]
                    tmp_dict={}
                    tmp_dict["segment"] = [st*frame_to_time/100.0, ed*frame_to_time/100.0]
                    tmp_dict["score"]= cls_anc[anc_idx][label]*1.0
                    tmp_dict["label"]=dataset.label_name[label]
                    tmp_dict["gentime"]= idx*frame_to_time/100.0
                    proposal_anc_dict.append(tmp_dict)
                          
            proposal_anc_dict = non_max_suppression(proposal_anc_dict, overlapThresh=opt['soft_nms'])  
                
            conf_queue[:-1,:]=conf_queue[1:,:].clone()
            conf_queue[-1,:]=0
            for proposal in proposal_anc_dict:
                cls_idx = dataset.label_name.index(proposal['label'])
                conf_queue[-1,cls_idx]=proposal["score"]
            
            minput = conf_queue.unsqueeze(0)
            suppress_conf = model(minput.cuda())
            suppress_conf=suppress_conf.squeeze(0).detach().cpu().numpy()
            
            for cls in range(0,num_class-1):
                if suppress_conf[cls] > opt['sup_threshold']:
                    for proposal in proposal_anc_dict:
                        if proposal['label'] == dataset.label_name[cls]:
                            if check_overlap_proposal(proposal_dict, proposal, overlapThresh=opt['soft_nms']) is None:
                                proposal_dict.append(proposal)
            
        result_dict[video_name]=proposal_dict
        proposal_dict=[]
        
    return result_dict

 
def test_frame(opt): 
    model = MYNET(opt).cuda()
    checkpoint = torch.load(opt["checkpoint_path"]+"/ckp_best.pth.tar")
    base_dict=checkpoint['state_dict']
    model.load_state_dict(base_dict)
    model.eval()
    
    dataset = VideoDataSet(opt,subset=opt['inference_subset'])    
    outfile = h5py.File(opt['frame_result_file'].format(opt['exp']), 'w')
    
    cls_loss, reg_loss, tot_loss, output_cls, output_reg, labels_cls, labels_reg, working_time, total_frames = eval_frame(opt, model,dataset)
    
    print("testing loss: %f, cls_loss: %f, reg_loss: %f"%(tot_loss, cls_loss, reg_loss ))
    
    for video_name in dataset.video_list:
        o_cls=output_cls[video_name]
        o_reg=output_reg[video_name]
        l_cls=labels_cls[video_name]
        l_reg=labels_reg[video_name]
        
        dset_predcls = outfile.create_dataset(video_name+'/pred_cls', o_cls.shape, maxshape=o_cls.shape, chunks=True, dtype=np.float32)
        dset_predcls[:,:] = o_cls[:,:]  
        dset_predreg = outfile.create_dataset(video_name+'/pred_reg', o_reg.shape, maxshape=o_reg.shape, chunks=True, dtype=np.float32)
        dset_predreg[:,:] = o_reg[:,:]   
        dset_labelcls = outfile.create_dataset(video_name+'/label_cls', l_cls.shape, maxshape=l_cls.shape, chunks=True, dtype=np.float32)
        dset_labelcls[:,:] = l_cls[:,:]   
        dset_labelreg = outfile.create_dataset(video_name+'/label_reg', l_reg.shape, maxshape=l_reg.shape, chunks=True, dtype=np.float32)
        dset_labelreg[:,:] = l_reg[:,:]   
    outfile.close()
                    
    print("working time : {}s, {}fps, {} frames".format(working_time, total_frames/working_time, total_frames))
    
def patch_attention(m):
    forward_orig = m.forward

    def wrap(*args, **kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False

        return forward_orig(*args, **kwargs)

    m.forward = wrap


class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out[1])

    def clear(self):
        self.outputs = []

def test(opt): 
    model = MYNET(opt).cuda()
    checkpoint = torch.load(opt["checkpoint_path"]+"/"+opt['exp']+"_ckp_best.pth.tar")
    base_dict=checkpoint['state_dict']
    model.load_state_dict(base_dict)
    model.eval()
    
    dataset = VideoDataSet(opt,subset=opt['inference_subset'])
    
    cls_loss, reg_loss, tot_loss, output_cls, output_reg, labels_cls, labels_reg, working_time, total_frames = eval_frame(opt, model,dataset)
    

    if opt["pptype"]=="nms":
        result_dict = eval_map_nms(opt,dataset, output_cls, output_reg, labels_cls, labels_reg)
    if opt["pptype"]=="net":
        result_dict = eval_map_supnet(opt,dataset, output_cls, output_reg, labels_cls, labels_reg)
    output_dict={"version":"VERSION 1.3","results":result_dict,"external_data":{}}
    outfile=open(opt["result_file"].format(opt['exp']),"w")
    json.dump(output_dict,outfile, indent=2)
    outfile.close()
    
    mAP = evaluation_detection(opt)


def test_online(opt): 
    model = MYNET(opt).cuda()
    checkpoint = torch.load(opt["checkpoint_path"]+"/ckp_best.pth.tar")
    base_dict=checkpoint['state_dict']
    model.load_state_dict(base_dict)
    model.eval()
    
    sup_model = SuppressNet(opt).cuda()
    checkpoint = torch.load(opt["checkpoint_path"]+"/ckp_best_suppress.pth.tar")
    base_dict=checkpoint['state_dict']
    sup_model.load_state_dict(base_dict)
    sup_model.eval()
    
    dataset = VideoDataSet(opt,subset=opt['inference_subset'])
    test_loader = torch.utils.data.DataLoader(dataset,
                                                batch_size=1, shuffle=False,
                                                num_workers=0, pin_memory=True,drop_last=False)
    
    result_dict={}
    proposal_dict=[]
    
    
    num_class = opt["num_of_class"]
    unit_size = opt['segment_size']
    threshold=opt['threshold']
    anchors=opt['anchors']
    
    start_time = time.time()
    total_frames =0 
    
    
    for video_name in dataset.video_list:
        input_queue = torch.zeros((unit_size,opt['feat_dim'])) 
        sup_queue = torch.zeros(((unit_size,num_class-1)))
    
        duration = dataset.video_len[video_name]
        video_time = float(dataset.video_dict[video_name]["duration"])
        frame_to_time = 100.0*video_time / duration
        
        for idx in range(0,duration):
            total_frames+=1
            input_queue[:-1,:]=input_queue[1:,:].clone()
            input_queue[-1:,:]=dataset._get_base_data(video_name,idx,idx+1)
            
            minput = input_queue.unsqueeze(0)
            act_cls, act_reg, _ = model(minput.cuda())
            act_cls = torch.softmax(act_cls, dim=-1)
            
            cls_anc = act_cls.squeeze(0).detach().cpu().numpy()
            reg_anc = act_reg.squeeze(0).detach().cpu().numpy()
            
            proposal_anc_dict=[]
            for anc_idx in range(0,len(anchors)):
                cls = np.argwhere(cls_anc[anc_idx][:-1]>opt['threshold']).reshape(-1)
                
                if len(cls) == 0:
                    continue
                    
                ed= idx + anchors[anc_idx] * reg_anc[anc_idx][0]
                length = anchors[anc_idx]* np.exp(reg_anc[anc_idx][1])
                st= ed-length
                
                for cidx in range(0,len(cls)):
                    label=cls[cidx]
                    tmp_dict={}
                    tmp_dict["segment"] = [st*frame_to_time/100.0, ed*frame_to_time/100.0]
                    tmp_dict["score"]= cls_anc[anc_idx][label]*1.0
                    tmp_dict["label"]=dataset.label_name[label]
                    tmp_dict["gentime"]= idx*frame_to_time/100.0
                    proposal_anc_dict.append(tmp_dict)
                          
            proposal_anc_dict = non_max_suppression(proposal_anc_dict, overlapThresh=opt['soft_nms'])  
                
            sup_queue[:-1,:]=sup_queue[1:,:].clone()
            sup_queue[-1,:]=0
            for proposal in proposal_anc_dict:
                cls_idx = dataset.label_name.index(proposal['label'])
                sup_queue[-1,cls_idx]=proposal["score"]
            
            minput = sup_queue.unsqueeze(0)
            suppress_conf = sup_model(minput.cuda())
            suppress_conf=suppress_conf.squeeze(0).detach().cpu().numpy()
            
            for cls in range(0,num_class-1):
                if suppress_conf[cls] > opt['sup_threshold']:
                    for proposal in proposal_anc_dict:
                        if proposal['label'] == dataset.label_name[cls]:
                            if check_overlap_proposal(proposal_dict, proposal, overlapThresh=opt['soft_nms']) is None:
                                proposal_dict.append(proposal)
            
        result_dict[video_name]=proposal_dict
        proposal_dict=[]
    
    end_time = time.time()
    working_time = end_time-start_time
    print("working time : {}s, {}fps, {} frames".format(working_time, total_frames/working_time, total_frames))
    
    output_dict={"version":"VERSION 1.3","results":result_dict,"external_data":{}}
    outfile=open(opt["result_file"].format(opt['exp']),"w")
    json.dump(output_dict,outfile, indent=2)
    outfile.close()
    
    evaluation_detection(opt)


def main(opt):
    max_perf=0
    if opt['mode'] == 'train':
        max_perf=train(opt)
    if opt['mode'] == 'test':
        test(opt)
    if opt['mode'] == 'test_frame':
        test_frame(opt)
    if opt['mode'] == 'test_online':
        test_online(opt)
    if opt['mode'] == 'eval':
        evaluation_detection(opt)
        
    return max_perf

if __name__ == '__main__':
    opt = opts.parse_opt()
    opt = vars(opt)
    if not os.path.exists(opt["checkpoint_path"]):
        os.makedirs(opt["checkpoint_path"]) 
    opt_file=open(opt["checkpoint_path"]+"/"+opt["exp"]+"_opts.json","w")
    json.dump(opt,opt_file)
    opt_file.close()
    
    if opt['seed'] >= 0:
        seed = opt['seed'] 
        torch.manual_seed(seed)
        np.random.seed(seed)
        #random.seed(seed)
           
    opt['anchors'] = [int(item) for item in opt['anchors'].split(',')]  
           
    main(opt)
    while(opt['wterm']):
        pass
