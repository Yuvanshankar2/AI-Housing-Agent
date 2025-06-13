from llama_cpp import Llama 

income = input("Enter income: ")
credit_score = input("Enter credit score: ")
ratio = input("Enter debt to income ratio: ")
numFamily = input("Enter number of family members: ")
llm = Llama(model_path="", n_ctx=2900,n_threads=4,n_gpu_layers=0)


prompt =  f"""You are a mathematically precise affordable rent calculator in the State of California. Your ONLY task is to calculate and display the affordable rent number using the following input and given MATHEMATICAL MODELS:

Input: 
1.  Income.
2.  Credit Score.
3.  Debt-to-income ratio
4.  Number of people in the family.

Models:
1. Multivariate Regression Analysis
2. Optimization models

Here is the income:
---
{income}
---
Here is the Credit Score:
---
{credit_score}
---
Here is the Debt-to-income ratio:
---
{ratio}
---
Here is the number of people in the family:
---
{numFamily}
---

Display ONLY the rent here:
--Give me the EXACT rent number here:
"""

res = llm(prompt, max_tokens=200)
print("-----------------------")
print(res["choices"][0]["text"])