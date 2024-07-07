import numpy as np
import h5py
import json
import torch
import torch.utils.data as data
import os
import pickle
from multiprocessing import Pool

def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data


def calc_iou(a, b):
    st = a[0]-a[1]
    ed = a[0]
    target_st = b[0]-b[1]
    target_ed = b[0]
    sst = min(st, target_st)
    led = max(ed, target_ed)
    lst = max(st, target_st)
    sed = min(ed, target_ed)

    iou = (sed-lst) / max(led-sst,1)
    return iou

def box_include(y, target): #is target is the larger box than y?
    st = y[0]-y[1]
    ed = y[0]
    target_st = target[0]-target[1]
    target_ed = target[0]
    
    detection_point = target_st #(target_st+target_ed)/2.0
    
    if ed > detection_point and target_st < st and target_ed > ed:
        return True
    return False    
                       
        
class VideoDataSet(data.Dataset):
    def __init__(self,opt,subset="train"):
        self.subset = subset
        self.mode = opt["mode"]
        self.predefined_fps = opt["predefined_fps"]
        self.video_anno_path = opt["video_anno"].format(opt["split"])          
        self.video_len_path = opt["video_len_file"].format(self.subset+'_'+opt["setup"])        
        self.num_of_class = opt["num_of_class"]
        self.segment_size = opt["segment_size"]
        self.label_name = []     
        self.match_score= {}       
        self.match_score_end= {}    
        self.match_length= {}    
        self.gt_action= {}
        self.cls_label={}
        self.reg_label={}
        self.snip_label={}
        self.inputs=[]
        self.inputs_all=[]
        self.data_rescale=opt["data_rescale"]
        self.anchors=opt["anchors"]
        self.pos_threshold=opt["pos_threshold"]
        
        self._getDatasetDict()  
        self._loadFeaturelen(opt)                  
        self._getMatchScore()
        self._makeInputSeq()
        self._loadPropLabel(opt['proposal_label_file'].format(self.subset+'_'+opt["setup"]))
        
        if self.subset == "train":
            if opt['data_format'] == "h5":
                feature_rgb_file = h5py.File(opt["video_feature_rgb_train"], 'r')
                self.feature_rgb_file={}
                keys = self.video_list
                for vidx in range(len(keys)):
                    self.feature_rgb_file[keys[vidx]]=np.array(feature_rgb_file[keys[vidx]][:])
                    
                if opt['rgb_only']:
                    self.feature_flow_file=None
                else:    
                    self.feature_flow_file={}
                    feature_flow_file = h5py.File(opt["video_feature_flow_train"], 'r')
                    for vidx in range(len(keys)):
                        self.feature_flow_file[keys[vidx]]=np.array(feature_flow_file[keys[vidx]][:])
            
            elif opt['data_format'] == "pickle":
                feature_All = pickle.load(open(opt["video_feature_all_train"], 'rb'))
                self.feature_rgb_file={}
                self.feature_flow_file={}
                
                keys = self.video_list
                for vidx in range(len(keys)):
                    self.feature_rgb_file[keys[vidx]]=feature_All[keys[vidx]]['rgb']#np.array(feature_All[keys[vidx]]['rgb'])
                for vidx in range(len(keys)):
                    self.feature_flow_file[keys[vidx]]=feature_All[keys[vidx]]['flow']#np.array(feature_All[keys[vidx]]['flow'])
            
            elif opt['data_format'] == "npz":
                feature_All = {}
                self.feature_rgb_file={}
                self.feature_flow_file={}
                for file in self.video_list:
                    feature_All[file] = np.load(opt["video_feature_all_train"]+file+'.npz')['feats']
                    
                keys = self.video_list
                for vidx in range(len(keys)):
                    self.feature_rgb_file[keys[vidx]]=feature_All[keys[vidx]][:]
                self.feature_flow_file = None

            elif opt['data_format'] == "npz_i3d":
                feature_All = {}
                self.feature_rgb_file={}
                self.feature_flow_file={}
                for file in self.video_list:
                    feature_All[file] = np.load(opt["video_feature_all_train"]+file+'.npz')
                    
                keys = self.video_list
                for vidx in range(len(keys)):
                    self.feature_rgb_file[keys[vidx]]=feature_All[keys[vidx]]['rgb']#np.array(feature_All[keys[vidx]]['rgb'])
                for vidx in range(len(keys)):
                    self.feature_flow_file[keys[vidx]]=feature_All[keys[vidx]]['flow']#np.array(feature_All[keys[vidx]]['flow'])

            elif opt['data_format'] == "pt":
                feature_All = {}
                self.feature_rgb_file={}
                self.feature_flow_file={}
                for file in self.video_list:
                    feature_All[file] = torch.load(opt["video_feature_all_train"]+file+'.pt')
                    
                keys = self.video_list
                for vidx in range(len(keys)):
                    self.feature_rgb_file[keys[vidx]]=feature_All[keys[vidx]][:]
                self.feature_flow_file = None
                            
        else:
            if opt['data_format'] == "h5":
                feature_rgb_file = h5py.File(opt["video_feature_rgb_test"], 'r')
                self.feature_rgb_file={}
                keys = self.video_list
                for vidx in range(len(keys)):
                    self.feature_rgb_file[keys[vidx]]=np.array(feature_rgb_file[keys[vidx]][:])
                    
                if opt['rgb_only']:
                    self.feature_flow_file=None
                else:    
                    self.feature_flow_file={}
                    feature_flow_file = h5py.File(opt["video_feature_flow_test"], 'r')
                    for vidx in range(len(keys)):
                        self.feature_flow_file[keys[vidx]]=np.array(feature_flow_file[keys[vidx]][:])
            
            elif opt['data_format'] == "pickle":
                feature_All = pickle.load(open(opt["video_feature_all_test"], 'rb') )
                self.feature_rgb_file={}
                self.feature_flow_file={}
                
                keys = self.video_list
                for vidx in range(len(keys)):
                    self.feature_rgb_file[keys[vidx]]=feature_All[keys[vidx]]['rgb']#np.array(feature_All[keys[vidx]]['rgb'])
                for vidx in range(len(keys)):
                    self.feature_flow_file[keys[vidx]]=feature_All[keys[vidx]]['flow']#np.array(feature_All[keys[vidx]]['flow']) 
            
            elif opt['data_format'] == "npz":
                self.feature_rgb_file={}
                self.feature_flow_file={}
                feature_All = {}
                for file in self.video_list:
                    feature_All[file] = np.load(opt["video_feature_all_test"]+file+'.npz')['feats']
                    # print(feature_All[file].shape)
                keys = self.video_list
                for vidx in range(len(keys)):
                    self.feature_rgb_file[keys[vidx]]=feature_All[keys[vidx]][:]
                self.feature_flow_file = None  

            elif opt['data_format'] == "npz_i3d":
                feature_All = {}
                self.feature_rgb_file={}
                self.feature_flow_file={}
                for file in self.video_list:
                    feature_All[file] = np.load(opt["video_feature_all_test"]+file+'.npz')
                    
                keys = self.video_list
                for vidx in range(len(keys)):
                    self.feature_rgb_file[keys[vidx]]=feature_All[keys[vidx]]['rgb']#np.array(feature_All[keys[vidx]]['rgb'])
                for vidx in range(len(keys)):
                    self.feature_flow_file[keys[vidx]]=feature_All[keys[vidx]]['flow']#np.array(feature_All[keys[vidx]]['flow'])

            elif opt['data_format'] == "pt":
                self.feature_rgb_file={}
                self.feature_flow_file={}
                feature_All = {}
                for file in self.video_list:
                    feature_All[file] = torch.load(opt["video_feature_all_test"]+file+'.pt')
                    # print(feature_All[file].shape, feature_All[file].dtype)
                keys = self.video_list
                for vidx in range(len(keys)):
                    self.feature_rgb_file[keys[vidx]]=feature_All[keys[vidx]][:]
                self.feature_flow_file = None    
    
    def _loadFeaturelen(self, opt):
        if os.path.exists(self.video_len_path):
            self.video_len = load_json(self.video_len_path)
            return
            
        self.video_len={}
        if self.subset == "train":
            if opt['data_format'] == "h5":
                feature_file = h5py.File(opt["video_feature_rgb_train"], 'r')
            elif opt['data_format'] == "pickle":
                feature_file = pickle.load(open(opt["video_feature_all_train"], 'rb'))
            elif opt['data_format'] == "npz":
                feature_file = {}
                for file in self.video_list:
                    feature_file[file] = np.load(opt["video_feature_all_train"]+file+'.npz')['feats']
            elif opt['data_format'] == "npz_i3d":
                feature_file = {}
                for file in self.video_list:
                    feature_file[file] = np.load(opt["video_feature_all_train"]+file+'.npz')
            elif opt['data_format'] == "pt":
                feature_file = {}
                for file in self.video_list:
                    feature_file[file] = torch.load(opt["video_feature_all_train"]+file+'.pt')
        else:
            if opt['data_format'] == "h5":
                feature_file = h5py.File(opt["video_feature_rgb_test"], 'r')
            elif opt['data_format'] == "pickle":
                feature_file = pickle.load(open(opt["video_feature_all_test"], 'rb'))
            elif opt['data_format'] == "npz":
                feature_file = {}
                for file in self.video_list:
                    feature_file[file] = np.load(opt["video_feature_all_test"]+file+'.npz')['feats']
            elif opt['data_format'] == "npz_i3d":
                feature_file = {}
                for file in self.video_list:
                    feature_file[file] = np.load(opt["video_feature_all_test"]+file+'.npz')
            elif opt['data_format'] == "pt":
                feature_file = {}
                for file in self.video_list:
                    feature_file[file] = torch.load(opt["video_feature_all_test"]+file+'.pt')
                    
                    
        keys = self.video_list
        if opt['data_format'] == "h5": 
            for vidx in range(len(keys)):
                self.video_len[keys[vidx]]=len(feature_file[keys[vidx]])
        elif opt['data_format'] == "pickle":
            for vidx in range(len(keys)):
                self.video_len[keys[vidx]]=len(feature_file[keys[vidx]]['rgb'])
        elif opt['data_format'] == "npz":
            for vidx in range(len(keys)):
                self.video_len[keys[vidx]]=len(feature_file[keys[vidx]])
        elif opt['data_format'] == "npz_i3d":
            for vidx in range(len(keys)):
                self.video_len[keys[vidx]]=len(feature_file[keys[vidx]]['rgb'])
        elif opt['data_format'] == "pt":
            for vidx in range(len(keys)):
                self.video_len[keys[vidx]]=len(feature_file[keys[vidx]])
        # print(self.video_len)
        outfile=open(self.video_len_path,"w")
        json.dump(self.video_len,outfile, indent=2)
        outfile.close()  
        
    def _getDatasetDict(self):
        anno_database= load_json(self.video_anno_path)
        anno_database=anno_database['database']
        self.video_dict = {}
        for video_name in anno_database:
            video_info=anno_database[video_name]
            video_subset=anno_database[video_name]['subset']
            if self.subset == "full":
                self.video_dict[video_name] = video_info
            if self.subset in video_subset:
                self.video_dict[video_name] = video_info
            
            for seg in video_info['annotations']:
                if not seg['label'] in self.label_name:
                    self.label_name.append(seg['label'])
        
        self.label_name.sort()            
        self.video_list = list(self.video_dict.keys())
        print ("%s subset video numbers: %d" %(self.subset,len(self.video_list)))
    
    def _getMatchScore(self):
        self.action_end_count = torch.zeros(2)
        for index in range(0, len(self.video_list)):
            video_name=self.video_list[index]
                            
            video_info=self.video_dict[video_name]
            video_labels=video_info['annotations']
            gt_bbox = []   
            gt_edlen = []   
            
            second_to_frame = self.video_len[video_name] / float(video_info['duration'])
            for j in range(len(video_labels)):
                tmp_info=video_labels[j]
                tmp_start= tmp_info['segment'][0]*second_to_frame
                tmp_end  = tmp_info['segment'][1]*second_to_frame
                tmp_label=self.label_name.index(tmp_info['label'])
                gt_bbox.append([tmp_start,tmp_end,tmp_label])
                gt_edlen.append([gt_bbox[-1][1], gt_bbox[-1][1]-gt_bbox[-1][0],tmp_label])
                              
            gt_bbox=np.array(gt_bbox)
            gt_edlen=np.array(gt_edlen)
            self.gt_action[video_name]=gt_edlen
            
            match_score=np.zeros((self.video_len[video_name],self.num_of_class-1), dtype=np.float32)
            for idx in range(gt_bbox.shape[0]):
                ed=int(gt_bbox[idx,1])+1
                st=int(gt_bbox[idx,0])
                match_score[st :ed, int(gt_bbox[idx,2])]=idx+1
            self.match_score[video_name]=match_score
            
    def _makeInputSeq(self):
        data_idx=0
        for index in range(0,len(self.video_list)):
            video_name=self.video_list[index]    
            #feature_rgb = self.feature_rgb_file[video_name]
            #feature_flow = self.feature_flow_file[video_name]
            #duration = min(len(feature_rgb), len(feature_flow))
            duration = self.match_score[video_name].shape[0]
            # print(video_name, duration)
            for i in range(1, duration+1):
                st = i-self.segment_size
                ed = i
                self.inputs_all.append([video_name,st,ed,data_idx])
                data_idx+=1
                
        self.inputs=self.inputs_all.copy()
        print ("%s subset seg numbers: %d" %(self.subset,len(self.inputs)))


    def _makePropLabelUnit(self, i):
        video_name=self.inputs_all[i][0]
        st = self.inputs_all[i][1]
        ed = self.inputs_all[i][2]
        # print(video_name)
        cls_anc=[]
        reg_anc=[]

        ### anchor annotation
        for j in range(0,len(self.anchors)):
            v1 = np.zeros(self.num_of_class)
            v1[-1]=1
            v2 = np.zeros(2)
            v2[-1]=-1e3
            # print(self.anchors[j])
            y_box = [ed-1, self.anchors[j]]
            
            subset_label=self._get_train_label_with_class(video_name,ed-self.anchors[j],ed)
            idx_list = []
            for ii in range(0, subset_label.shape[0]):
                for jj in range(0, subset_label.shape[1]):
                    idx=int(subset_label[ii,jj])
                    if idx>0 and idx-1 not in idx_list:
                        idx_list.append(idx-1)
            
            for idx in idx_list:
                target_box = self.gt_action[video_name][idx]
                cls = int(target_box[2])
                iou = calc_iou(y_box,target_box)
                if iou >= self.pos_threshold or (j == len(self.anchors)-1 and box_include(y_box, target_box)) or (j==0 and box_include(target_box, y_box)):
                    v1[cls]=1
                    v1[-1]=0
                    v2[0]=1.0*(target_box[0]-y_box[0])/self.anchors[j]
                    v2[1]=np.log(1.0*max(1,target_box[1])/y_box[1])
            
            cls_anc.append(v1)
            reg_anc.append(v2)

        ### snippet level annotation
        v0 = np.zeros(self.num_of_class)
        v0[-1]=1
        segment_size = ed - st
        y_box = [ed-1, self.anchors[-1]]

        subset_label=self._get_train_label_with_class(video_name,ed-self.anchors[-1],ed)
        idx_list = []
        for ii in range(0, subset_label.shape[0]):
            for jj in range(0, subset_label.shape[1]):
                idx=int(subset_label[ii,jj])
                if idx>0 and idx-1 not in idx_list:
                    idx_list.append(idx-1)
        
        for idx in idx_list:
            target_box = self.gt_action[video_name][idx]
            cls = int(target_box[2])
            iou = calc_iou(y_box,target_box)
            if iou >= 0: #any overlapping
                v0[cls]=1
                v0[-1]=0

        cls_anc=np.stack(cls_anc, axis=0)
        reg_anc=np.stack(reg_anc, axis=0)
        cls_snip = np.array(v0)
        return cls_anc,reg_anc,cls_snip
    
    def _loadPropLabel(self, filename):
        if os.path.exists(filename):
            prop_label_file = h5py.File(filename, 'r')
            self.cls_label=np.array(prop_label_file['cls_label'][:])
            self.reg_label=np.array(prop_label_file['reg_label'][:])
            self.snip_label=np.array(prop_label_file['snip_label'][:])
            prop_label_file.close()
            self.action_frame_count = np.sum(self.cls_label.reshape((-1,self.cls_label.shape[-1])),axis=0)
            self.action_frame_count=torch.Tensor(self.action_frame_count)
            return
    
        pool = Pool(os.cpu_count()//2)
        labels = pool.map(self._makePropLabelUnit, range(0,len(self.inputs_all)))
        pool.close()
        pool.join()
        
        cls_label=[]
        reg_label=[]
        snip_label=[]
        for i in range(0,len(labels)):
            cls_label.append(labels[i][0])
            reg_label.append(labels[i][1])
            snip_label.append(labels[i][2])
        self.cls_label=np.stack(cls_label,axis=0)
        self.reg_label=np.stack(reg_label,axis=0)
        self.snip_label=np.stack(snip_label,axis=0)
        
        outfile = h5py.File(filename, 'w')
        dset_cls = outfile.create_dataset('/cls_label', self.cls_label.shape, maxshape=self.cls_label.shape, chunks=True, dtype=np.float32)
        dset_cls[:,:] = self.cls_label[:,:]  
        dset_reg = outfile.create_dataset('/reg_label', self.reg_label.shape, maxshape=self.reg_label.shape, chunks=True, dtype=np.float32)
        dset_reg[:,:] = self.reg_label[:,:]  
        dset_snip = outfile.create_dataset('/snip_label', self.snip_label.shape, maxshape=self.snip_label.shape, chunks=True, dtype=np.float32)
        dset_snip[:,:] = self.snip_label[:,:]  
        outfile.close()
        
        return  
                

    def __getitem__(self, index):
        video_name, st, ed, data_idx = self.inputs[index]        
        if st >= 0:
            feature = self._get_base_data(video_name,st,ed)
        else :
            feature = self._get_base_data(video_name,0,ed)
            padfunc2d = torch.nn.ConstantPad2d((0,0,-st,0), 0)
            feature=padfunc2d(feature)
        
        #match_score = torch.Tensor(self.match_score[video_name][ed-1])
        #match_score =  self._get_train_label_with_class(video_name,st,ed)
        
        cls_label=torch.Tensor(self.cls_label[data_idx])
        reg_label=torch.Tensor(self.reg_label[data_idx])
        snip_label = torch.Tensor(self.snip_label[data_idx])
        # print(cls_label[-1], snip_label)
            
        return feature,cls_label,reg_label,snip_label
        
            
    def _get_base_data(self,video_name,st,ed): 
        feature_rgb = self.feature_rgb_file[video_name]
        feature_rgb = feature_rgb[st:ed,:]
        
        if self.feature_flow_file is not None:
            feature_flow = self.feature_flow_file[video_name]
            # print(feature_flow.shape)
            feature_flow = feature_flow[st:ed,:]
            feature = np.append(feature_rgb,feature_flow, axis=1)
        else:
            feature = feature_rgb
        feature = torch.from_numpy(np.array(feature))
     
        return feature           
    
    def _get_train_label_with_class(self,video_name,st,ed): 
        duration=len(self.match_score[video_name])
        st_padding=0
        ed_padding=0
        if st<0:
            st_padding=-st
            st=0
        if ed > duration:
            ed_padding = ed - duration
            ed = duration
    
        match_score = torch.Tensor(self.match_score[video_name][st:ed])
        if st_padding > 0:
            padfunc2d = torch.nn.ConstantPad2d((0,0,st_padding,0),0)
            match_score = padfunc2d(match_score)
        if ed_padding > 0:
            padfunc2d = torch.nn.ConstantPad2d((0,0,0,ed_padding),0)
            match_score = padfunc2d(match_score)
        return match_score
    
    def __len__(self):
        return len(self.inputs)
    
    def reset_sample(self):
        self.inputs = self.inputs_all.copy()
        
    def select_sample(self,idx):
        inputs = [self.inputs_all[i] for i in idx]
        self.inputs = inputs.copy()
        return
        

class SuppressDataSet(data.Dataset):
    def __init__(self,opt,subset="train"):
        self.subset = subset
        self.mode = opt["mode"]
        self.data_file = h5py.File(opt["suppress_label_file"].format(self.subset+"_"+opt['setup']), 'r')
        self.video_list = list(self.data_file.keys())
        self.inputs=[]
        for index in range(0,len(self.video_list)):
            video_name=self.video_list[index]
            duration = self.data_file[video_name+'/input'].shape[0]
            for i in range(0, duration):
                self.inputs.append([video_name,i])
                
        print ("%s subset seg numbers: %d" %(self.subset,len(self.inputs)))
        
    def __getitem__(self, index):
        video_name, idx = self.inputs[index]     
        
        input_seq = self.data_file[video_name+'/input'][idx]
        label = self.data_file[video_name+'/label'][idx]
        
        input_seq= torch.from_numpy(input_seq)
        label = torch.from_numpy(label)
        
        return input_seq, label
            
    def __len__(self):
        return len(self.inputs)
            
