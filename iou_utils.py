import numpy as np

def non_max_suppression(proposals, overlapThresh=0.3):
    # if there are no intervals, return an empty list
    if len(proposals) == 0:
        return []

    # initialize the list of picked indexes
    pick = []
    
    sorted_proposal = sorted(proposals, key=lambda proposal:proposal['score'], reverse=True)
    idx=0
    total_proposal= len(sorted_proposal)
    while idx < total_proposal: 
        proposal = sorted_proposal[idx]
        st = proposal['segment'][0]
        ed = proposal['segment'][1]
        label = proposal['label']
        
        delete_item = []
        for j in range(idx+1, total_proposal):
            target_proposal = sorted_proposal[j]
            target_st = target_proposal['segment'][0]
            target_ed = target_proposal['segment'][1]
            target_label = target_proposal['label']
            
            if(label == target_label):
                sst = np.minimum(st, target_st)
                led = np.maximum(ed, target_ed)
                lst = np.maximum(st, target_st)
                sed = np.minimum(ed, target_ed)
                
                iou = (sed-lst) / max(led-sst,1)
                if(iou > overlapThresh):
                    delete_item.append(target_proposal)
                    
        for item in delete_item:
            sorted_proposal.remove(item)
        total_proposal=len(sorted_proposal)
        idx+=1
        
    return sorted_proposal
    
    
def check_overlap_proposal(proposal_list, new_proposal, overlapThresh=0.3):
    for proposal in proposal_list:
        st = proposal['segment'][0]
        ed = proposal['segment'][1]
        label = proposal['label']
        
        new_st = new_proposal['segment'][0]
        new_ed = new_proposal['segment'][1]
        new_label = new_proposal['label']
        
        if(label == new_label):
            sst = np.minimum(st, new_st)
            led = np.maximum(ed, new_ed)
            lst = np.maximum(st, new_st)
            sed = np.minimum(ed, new_ed)
            
            iou = (sed-lst) / max(led-sst,1)
            if(iou > overlapThresh):
                return proposal
    
    return None
