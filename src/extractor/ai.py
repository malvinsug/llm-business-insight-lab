class AITextExtractor:
    """
    Extract facts and recomendations from AI-generated business insight text. 
    
    NOTE: We assume that that the business insights elements (observations, hypotheses and recommendations) from the reasoning evaluation from LLM is complete.
    """

    @staticmethod
    def extract_observations_and_hypotheses(reasoning_dict:list) -> list:
        if isinstance(reasoning_dict,list):
            observation_list = [obs for item in reasoning_dict for obs in item.get("observations", [])]
            hypothesis_list = []
            
            for item in reasoning_dict:
                if isinstance(item, str):
                    hypothesis_list.append(item)
                elif isinstance(item, list):
                    for hyp in item:
                        hypothesis_list.append(hyp)

            return list(set(observation_list + hypothesis_list))

        return list(set(reasoning_dict["observations"] + reasoning_dict["hypotheses"]))
    
    @staticmethod
    def extract_recommendations(reasoning_dict:list)->list:
        if isinstance(reasoning_dict,list):
            return [obj["recommendations"] for obj in reasoning_dict]

        return [reasoning_dict["recommendations"]]
