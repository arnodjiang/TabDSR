import evaluate


rouge = evaluate.load('/raid/share/jiangchangjiang/tablellmPipeline/evaluates/rouge')
# predictions = ["hello there", "general kenobi"]
# references = ["hello there", "general kenobi general"]
predictions = ["general kenobi general"]
references = ["general kenobi general"]
results = rouge.compute(predictions=predictions, references=references)
print(results)