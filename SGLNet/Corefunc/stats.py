# -*- coding: utf-8 -*-
"""
Created on Mon May  5 13:30:18 2025

@author: niels
"""

import numpy as np
import numpy.typing as npt

def TP(pred: npt.NDArray, tgt: npt.NDArray) -> npt.NDArray[bool]:
    return ((pred == 1) & (tgt == 1))

def TN(pred: npt.NDArray, tgt: npt.NDArray) -> npt.NDArray[bool]:
    return ((pred == 0) & (tgt == 0))

def FP(pred: npt.NDArray, tgt: npt.NDArray) -> npt.NDArray[bool]:
    return ((pred == 1) & (tgt == 0))

def FN(pred: npt.NDArray, tgt: npt.NDArray) -> npt.NDArray[bool]:
    return ((pred == 0) & (tgt == 1))

def sum_stats(elements: tuple) -> tuple[float]:
    return tuple(sum(elm) for elm in elements)

def arr_accuracy(tp: npt.NDArray, tn: npt.NDArray, fp: npt.NDArray, fn: npt.NDArray) -> float:
    tp, tn, fp, fn = sum_stats((tp, tn, fp, fn))
    return (tp+tn) / (tp+tn+fp+fn)

def arr_specificity(tn: npt.NDArray, fp: npt.NDArray) -> float:
    tn, fp = sum_stats((tn, fp))
    return tn / (tn+fp)

def arr_precision(tp: npt.NDArray, fp: npt.NDArray) -> float:
    tp, fp = sum_stats((tp, fp))
    return tp / (tp+fp)

def arr_recall(tp: npt.NDArray, fn: npt.NDArray) -> float:
    tp, fn = sum_stats((tp, fn))
    return tp / (tp+fn)

def arr_f1(tp: npt.NDArray, fp: npt.NDArray, fn:npt.NDArray) -> float:
    prec = precision(tp, fp)
    rec = recall(tp, fn)
    return (2*prec*rec) / (prec+rec)

def accuracy(tp: int, tn: int, fp: int, fn: int) -> int:
    return (tp+tn) / (tp+tn+fp+fn)

def specificity(tn: int, fp: int) -> int:
    return tn / (tn+fp)

def precision(tp: int, fp: int) -> int:
    return tp / (tp+fp)

def recall(tp: int, fn: int) -> int:
    return tp / (tp+fn)

def f1(tp: int, fp: int, fn:int) -> int:
    prec = precision(tp, fp)
    rec = recall(tp, fn)
    return (2*prec*rec) / (prec+rec)

def union(seg: npt.NDArray, tgt: npt.NDArray) -> npt.NDArray:
    return seg | tgt

def intersect(seg: npt.NDArray, tgt: npt.NDArray) -> npt.NDArray:
    return seg & tgt

def dice_coef(size_A: int, size_B: int, size_intersect: int) -> float:
    num = 2*size_intersect
    den = size_A+size_B
    if den > 0:
        return num/den
    return None

def jaccard_index(size_intersect: int, size_union: int) -> float:
    if size_union > 0:
        return size_intersect / size_union
    return None


class ConfusionMatrix:
    
    def __init__(self, pred: npt.NDArray, tgt: npt.NDArray) -> None:
        self.update(pred, tgt)
        
    def update(self, pred: npt.NDArray, tgt: npt.NDArray) -> None:
        self.pred_arr = pred
        self.tgt_arr = tgt
        self.tp_arr = TP(pred, tgt)
        self.tn_arr = TN(pred, tgt)
        self.fp_arr = FP(pred, tgt)
        self.fn_arr = FN(pred, tgt)
        self.tp = int(sum(self.tp_arr))                                                                                                       
        self.tn = int(sum(self.tn_arr))
        self.fp = int(sum(self.fp_arr))
        self.fn = int(sum(self.fn_arr))
        
    def __call__(self) -> dict:
        return {
            'tp': self.tp,
            'tn': self.tn,
            'fp': self.fp,
            'fn': self.fn
        }
    
    
class ClassificationMetrics:
    
    def __init__(self) -> None:
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        
    def update(self, tp: int, tn: int, fp: int, fn: int) -> None:
        self.tp += int(tp)                                                                                                       
        self.tn += int(tn)
        self.fp += int(fp)
        self.fn += int(fn)
        
    def __call__(self) -> dict:
        return {
            'tp': self.tp,
            'tn': self.tn,
            'fp': self.fp,
            'fn': self.fn,
            'f1': f1(self.tp, self.fp, self.fn),
            'accu': accuracy(self.tp, self.tn, self.fp, self.fn),
            'spec': specificity(self.tn, self.fp),
            'prec': precision(self.tp, self.fp),
            'reca': recall(self.tp, self.fn),
        }
    
    
class SegmentationOverlap:
    
    def __init__(self, seg: npt.NDArray, tgt: npt.NDArray) -> None:
        self.update(seg, tgt)
        
    def update(self, seg: npt.NDArray, tgt: npt.NDArray) -> None:
        self.seg = seg.astype(bool)
        self.tgt = tgt.astype(bool)
        self.union_mask = union(self.seg, self.tgt)
        self.intersect_mask = intersect(self.seg, self.tgt)
        self.union = int(np.sum(self.union_mask))
        self.intersect = int(np.sum(self.intersect_mask))
        self.seg_size = int(np.sum(self.seg))
        self.tgt_size = int(np.sum(self.tgt))
        self.dice = dice_coef(self.seg_size, self.tgt_size, self.intersect)
        self.jaccard = jaccard_index(self.intersect, self.union)
        
    def __call__(self) -> dict:
        return {
            'seg_size': self.seg_size,
            'tgt_size': self.tgt_size,
            'intersect': self.intersect,
            'union': self.union,
            'dice': self.dice,
            'jaccard': self.jaccard
        }
    
    
class SegmentationMetrics:
    
    def __init__(self) -> None:
        self.seg_size = 0
        self.tgt_size = 0
        self.intersect = 0
        self.union = 0
        self.dice = 0
        self.jaccard = 0
        self.count = 0
        
    def update(self, seg_size: int, tgt_size: int, intersect: int, union: int, dice: float, jaccard: float) -> None:
        self.seg_size += int(seg_size)
        self.tgt_size += int(tgt_size)
        self.intersect += int(intersect)
        self.union += int(union)
        self.dice += float(dice)                                                                                  
        self.jaccard += float(jaccard)
        self.count += 1
        
    def __call__(self) -> dict:
        return {
            'seg_size': self.seg_size,
            'tgt_size': self.tgt_size,
            'intersect': self.intersect,
            'union': self.union,
            'count': self.count,
            'dice': self.dice,
            'jaccard': self.jaccard,
            'mean_dice': self.dice/self.count,
            'mean_jaccard': self.jaccard/self.count,
            'global_dice': dice_coef(self.seg_size, self.tgt_size, self.intersect),
            'global_jaccard': jaccard_index(self.intersect, self.union)
        }