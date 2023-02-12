from simpletransformers.question_answering import QuestionAnsweringModel
import json
import logging
import os
if __name__ == '__main__':

    NUM_EPOCHS = 10

    with open('train.json') as f:
        train = json.load(f)

    with open('test.json') as f:
        test = json.load(f)

    model_type = 'bert'
    model_name = './model'
    os.system('clear')

    ### Advanced Methodology
    """train_args = {
        "reprocess_input_data": True,
        "overwrite_output_dir": True,
        "use_cached_eval_features": True,
        "output_dir": f"outputs/{model_type}",
        "best_model_dir": f"outputs/{model_type}/best_model",
        "evaluate_during_training": True,
        "max_seq_length": 128,
        "num_train_epochs": 10,
        "evaluate_during_training_steps": 1000,
        "save_model_every_epoch": False,
        "save_eval_checkpoints": False,
        "n_best_size":3,
        # "use_early_stopping": True,
        # "early_stopping_metric": "mcc",
        # "n_gpu": 2,
        # "manual_seed": 4,
        "use_multiprocessing": False,
        "train_batch_size": 128,
        "eval_batch_size": 64,
        # "config": {
        #     "output_hidden_states": True
        # }
    }"""   
    
    args = {
        "n_best_size":3,
        "num_train_epochs": 10,
        "best_model_dir": f"outputs/{model_type}/best_model",
        "output_dir": f"outputs/{model_type}",
        "evaluate_during_training_steps": 1000,
        "evaluate_during_training": True,
    }

    model = QuestionAnsweringModel(model_type, model_name, args=args, use_cuda=False)


    model.train_model(train, eval_data=test)
    result, texts = model.eval_model(test)
    print(result)