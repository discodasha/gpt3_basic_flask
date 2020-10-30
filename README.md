# gpt3_basic_flask


example of API call

curl -H "Content-Type: application/json" -d '{"prompt":"Конь", "length":10,"temperature":1.0, "top_k": 5, "top_p":0.95, "repetition_penalty":3.0,"num_return_sequences":1}' -X POST http:<host>:<port>/gpt3/get


if you start app for the first time, use model_name_or_path = "sberbank-ai/rugpt3large_based_on_gpt2" to load model
