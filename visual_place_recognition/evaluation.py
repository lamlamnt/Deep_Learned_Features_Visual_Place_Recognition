class Evaluation:
    #maybe includes config
    def __init__(self,reference_run,query_run,reference_len,query_len,increment_ref,increment_que):
        self.reference_run = reference_run
        self.query_run = query_run
        self.reference_len = reference_len
        self.query_len = query_len
        self.increment_ref = increment_ref
        self.increment_que = increment_que
    
    #Set similarity_run, distance_error_frame, ref_gps, query_gps, max_similarity_idx, localized_frame