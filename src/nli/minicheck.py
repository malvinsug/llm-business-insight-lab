from bespokelabs import BespokeLabs
from minicheck.minicheck import MiniCheck
import os

from src.utils.config import BESPOKE_API_KEY

class NLIMiniCheckEvaluator:
    """
    NLI-wrapper class which evaluates the factuality of a text based on a reference. 
    """

    BESPOKE_API_KEY = BESPOKE_API_KEY

    @staticmethod
    def get_nli_model():
        if NLIMiniCheckEvaluator.BESPOKE_API_KEY !="":
            return BespokeLabs(auth_token= NLIMiniCheckEvaluator.BESPOKE_API_KEY)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            return MiniCheck(model_name='deberta-v3-large', cache_dir='./ckpts')

    TH_ALIGN_LIST = [0.10,0.20,0.25,0.30,0.35,0.40,0.45,0.50]

    @staticmethod
    def label_claim_minicheck(nli_model: BespokeLabs | MiniCheck,
                        claim: str,
                        document_pool: list[str],
                        k: int = 5,
                        th_align_list: list[float] = TH_ALIGN_LIST
                       ) -> tuple[list[str], list[float], list[str]]:
        
        best_p, best_evid = -1.0, ""
        
        if len(document_pool) == 1:
            premises = document_pool
        else:
            premises = document_pool[:k]

        if isinstance(nli_model, BespokeLabs):
            for prem in premises:
                resp = nli_model.minicheck.factcheck.create(
                    claim=claim,
                    context=prem
                )
                nli_score_probability = resp.support_prob

                if nli_score_probability > best_p:
                    best_p, best_evid = nli_score_probability, prem
        
        elif isinstance(nli_model, MiniCheck):
            for prem in premises:
                nli_score_probability = nli_model.score(docs = [prem], claims=[claim])[1][0] 


                if nli_score_probability > best_p:
                    best_p, best_evid = nli_score_probability, prem


        # Decide label based on threshold
        best_label_list = []
        best_prob_list = []
        best_evidence_list = []
        for th_align in th_align_list:
            if best_p >= th_align:
                best_lbl = "ALIGN"

            else:
                best_lbl = "NOT_ALIGN"

            best_label_list.append(best_lbl)
            best_prob_list.append(best_p)
            best_evidence_list.append(best_evid)

        return best_label_list, best_prob_list, best_evidence_list

    @staticmethod
    def evaluate_answer_minicheck(
        nli_model: BespokeLabs | MiniCheck,
        claims_to_check: list[str], 
        context_document:str, 
        k: int, 
        recommendation_or_fact_str:str
    ) -> dict:
        
        """
        Produces fact_check_dict
        """
        
        per_claim = []
        if any(isinstance(claim, list) for claim in claims_to_check):
            # unnest claims_to_check
            unnested_claims_to_check = []
            for sublist in claims_to_check:
                if isinstance(sublist, str):
                    unnested_claims_to_check.append(sublist)
                elif isinstance(sublist, list):
                    for item in sublist:
                        unnested_claims_to_check.append(item)
            claims_to_check = unnested_claims_to_check
            print(claims_to_check)

        for c in claims_to_check:
            try:
                assert isinstance(c, str) and len(c) > 1
            except:
                print(claims_to_check)
            try:
                best_label_list, best_prob_list, best_evidence_list = NLIMiniCheckEvaluator.label_claim_minicheck(nli_model, c, [context_document], k=k)
            except:
                #TODO: Refine this Exception. It was only for debugging back then.
                print(best_label_list, best_prob_list, best_evidence_list)
            
            p = best_prob_list[0]
            ev = best_evidence_list[0]
            
            per_claim.append({
                "claim":    c,
                "label_10":    best_label_list[0],
                "label_20":    best_label_list[1],
                "label_25":    best_label_list[2],
                "label_30":    best_label_list[3],
                "label_35":    best_label_list[4],
                "label_40":    best_label_list[5],
                "label_45":    best_label_list[6],
                "label_50":    best_label_list[7],
                "p":        round(p, 3),
                "evidence": ev
            })

        # Aggregate counts by whatever labels we saw
        labels_10 = {r["label_10"] for r in per_claim}
        labels_20 = {r["label_20"] for r in per_claim}
        labels_25 = {r["label_25"] for r in per_claim}
        labels_30 = {r["label_30"] for r in per_claim}
        labels_35 = {r["label_35"] for r in per_claim}
        labels_40 = {r["label_40"] for r in per_claim}
        labels_45 = {r["label_45"] for r in per_claim}
        labels_50 = {r["label_50"] for r in per_claim}

        summary_10 = {lab: sum(r["label_10"] == lab for r in per_claim) for lab in labels_10}
        summary_20 = {lab: sum(r["label_20"] == lab for r in per_claim) for lab in labels_20}
        summary_25 = {lab: sum(r["label_25"] == lab for r in per_claim) for lab in labels_25}
        summary_30 = {lab: sum(r["label_30"] == lab for r in per_claim) for lab in labels_30}
        summary_35 = {lab: sum(r["label_35"] == lab for r in per_claim) for lab in labels_35}
        summary_40 = {lab: sum(r["label_40"] == lab for r in per_claim) for lab in labels_40}
        summary_45 = {lab: sum(r["label_45"] == lab for r in per_claim) for lab in labels_45}
        summary_50 = {lab: sum(r["label_50"] == lab for r in per_claim) for lab in labels_50}

        summary_10["total"] = len(per_claim)
        summary_20["total"] = len(per_claim)
        summary_25["total"] = len(per_claim)
        summary_30["total"] = len(per_claim)
        summary_35["total"] = len(per_claim)
        summary_40["total"] = len(per_claim)
        summary_45["total"] = len(per_claim)
        summary_50["total"] = len(per_claim)




        return {
            f"{recommendation_or_fact_str}_10_summary":   summary_10,
            f"{recommendation_or_fact_str}_20_summary":   summary_20,
            f"{recommendation_or_fact_str}_25_summary":   summary_25,
            f"{recommendation_or_fact_str}_30_summary":   summary_30,
            f"{recommendation_or_fact_str}_35_summary":   summary_35,
            f"{recommendation_or_fact_str}_40_summary":   summary_40,
            f"{recommendation_or_fact_str}_45_summary":   summary_45,
            f"{recommendation_or_fact_str}_50_summary":   summary_50,
            f"{recommendation_or_fact_str}_per_claim": per_claim
        }
    
    @staticmethod
    def calculate_factuality_precision(fact_check_dict:dict, prefix:str):
        try:
            return 1 - fact_check_dict[f"{prefix}_summary"]["NOT_ALIGN"] / fact_check_dict[f"{prefix}_summary"]["total"]
        except:
            return fact_check_dict[f"{prefix}_summary"]["ALIGN"] / fact_check_dict[f"{prefix}_summary"]["total"]