# Task: Oscillator1 method EoH

from llm4ad.task.optimization.online_bin_packing import OBPEvaluation
from llm4ad.task.science_discovery.oscillator1 import OscillatorEvaluation1
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.method.eoh import EoH, EoHProfiler
import os
apikey = os.getenv('DEEPSEEK_API_KEY')
print(apikey)
if __name__ == '__main__':
    llm = HttpsApi(
        host='api.deepseek.com',   # your host endpoint, e.g., api.openai.com, api.deepseek.com
        key=apikey, # your key, e.g., sk-xxxxxxxxxx
        model='deepseek-chat',  # your llm, e.g., gpt-3.5-turbo, deepseek-chat
        timeout=50
    )
    task = OscillatorEvaluation1()
    method = EoH(
        llm=llm,
        profiler=EoHProfiler(log_dir='logs/eoh', log_style='simple'),
        evaluation=task,
        max_sample_nums=20,
        max_generations=10,
        pop_size=4,
        num_samplers=1,
        num_evaluators=1,
        debug_mode=False
    )
    method.run()