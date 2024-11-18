import evaluate
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def score(input_texts : list[str], model_id : str, add_start_token : bool) -> dict:
    """
    Wrapper fn written to take a list of strings, computes perplexity score for strings passed.
    """
    try:
        perplexity = evaluate.load("perplexity", module_type="metric")
        results = perplexity.compute(model_id=model_id,
                                    add_start_token=add_start_token,
                                    predictions=input_texts)
        return {
            "success" : True,
            "data" : results,
            "error" : None
        }
    except Exception as e:
        logging.error(f"score fn could not compute perplexity due to {str(e)}")
        return {
            "success" : False,
            "data" : None,
            "error" : str(e)
        }
