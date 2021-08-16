import numpy as np
import torch

class PieceWeight:
    def __init__(self, piece, weight):
        self.piece = piece
        self.weight = weight


class RecognizeResult:
    def __init__(self):
        self.ret_flag = -1
        self.result_item_list = []

class ResultItem:
    def __init__(self):
        self.score = 0.0
        self.text = ""
        self.piece_weight_list = []


class Recognizer:
    def __init__(self,labels,feature,model):
        self.labels =  labels
        self.feature = feature
        self.model = model 
    
    def predict_given_feature(self, audio_feature, feature_length, beam_size=1, nbest=1):

        result = RecognizeResult()
        
        with torch.no_grad():
            nbest_hyps = self.model.recognize(audio_feature,  
                                            feature_length,
                                            beam_size,        
                                            nbest,           
                                            100,
                                            self.labels,
                                            verbose=False
                                            )

        result_item_list = self._parse_hyps(nbest_hyps)
        
        if result_item_list is None:
            return result

        result.result_item_list = result_item_list
        result.ret_flag = 0
        return result

    def _parse_hyps(self, nbest_hyps):
        result_item_list = []
        for hyp in nbest_hyps:
            result_item = ResultItem()
            yseq = hyp['yseq']
            word_confidences = hyp['word_confidences']
            result_item.score = np.mean(word_confidences)

            if len(word_confidences) != len(yseq):
                return None

            text_pieces = []
            for idx in range(len(yseq)):
                tokenid = yseq[idx]
                if tokenid in (self.model.decoder.sos_id, self.model.decoder.eos_id):
                    continue
                text_piece = self.labels.tokens_to_text([tokenid])
                text_pieces.append(text_piece)
                result_item.piece_weight_list.append(PieceWeight(text_piece, word_confidences[idx]))
            result_item.text = "".join(text_pieces)
            result_item_list.append(result_item)
        return result_item_list



