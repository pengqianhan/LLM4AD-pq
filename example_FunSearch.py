# Task: Oscillator1 method EoH

from llm4ad.task.science_discovery.oscillator1 import OscillatorEvaluation1
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.tools.llm.llm_api_openai import OpenAIAPI
from llm4ad.method.funsearch import FunSearch
from llm4ad.method.funsearch.profiler import FunSearchProfiler
import os
from dotenv import load_dotenv
load_dotenv()
apikey = os.getenv('GEMINI_API_KEY')
if __name__ == '__main__':
    # llm = HttpsApi(
    #     host='api.deepseek.com',   # your host endpoint, e.g., api.openai.com, api.deepseek.com
    #     key=apikey, # your key, e.g., sk-xxxxxxxxxx
    #     model='deepseek-chat',  # your llm, e.g., gpt-3.5-turbo, deepseek-chat
    #     timeout=100
    # )
    llm = OpenAIAPI(
        base_url='https://generativelanguage.googleapis.com/v1beta/openai/',
        api_key=apikey,
        model='models/gemini-flash-latest',
        timeout=100
    )
    task = OscillatorEvaluation1()
    method = FunSearch(
        llm=llm,
        profiler=FunSearchProfiler(log_dir='logs/funsearch', log_style='simple'),
        evaluation=task,
        max_sample_nums=20,
        max_generations=10,
        pop_size=4,
        num_samplers=1,
        num_evaluators=1,
        debug_mode=False
    )
    method.run()