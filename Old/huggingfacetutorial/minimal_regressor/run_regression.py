# source: https://gist.github.com/adfindlater/be8d3dfedb0361cae381a240f1022938

# coding=utf-8
#
# Minimal BERT regression model
#
# Code borrowed from huggingface text-classification example (original licence follows):
# https://github.com/huggingface/transformers/blob/master/examples/text-classification/run_glue.py
#
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""

# requirements.txt:
# datasets
# transformers

# RUN SCRIPT:
# export MAX_LENGTH=128
# export LEARNING_RATE=2e-5
# export BATCH_SIZE=32
# export NUM_EPOCHS=3
# export SEED=2
# export OUTPUT_DIR_NAME=bert-output
# export CURRENT_DIR=${PWD}

# export BERT_MODEL=bert-base-cased
# export TASK=mrpc
# export DATA_DIR=./glue_data/MRPC/
# export OUTPUT_DIR=${CURRENT_DIR}/${OUTPUT_DIR_NAME}

# mkdir -p $OUTPUT_DIR
# export PYTHONPATH="../":"${PYTHONPATH}"

# python3.6 run_glue.py --data_dir $CURRENT_DIR \
# --task $TASK \
# --model_name_or_path $BERT_MODEL \
# --output_dir $OUTPUT_DIR \
# --do_train \
# --do_eval \
# --do_predict \
# --max_seq_length  $MAX_LENGTH \
# --learning_rate $LEARNING_RATE \
# --num_train_epochs $NUM_EPOCHS \
# --seed $SEED \


# import dataclasses
import logging
import os
import sys

from typing import Optional
from dataclasses import dataclass, field

from datasets import load_dataset

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    num_labels = 1
    
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # Get datasets
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    
    train_dataset = load_dataset("csv", data_files="train.csv")
    train_dataset = train_dataset.map(
        lambda e: tokenizer(e["data_LossDescription"], padding=True, truncation=True, max_length=128), batched=True
    )
    train_dataset.set_format(
        type="torch",
        columns=["input_ids", "token_type_ids", "attention_mask", "labels"],
    )

    eval_dataset = load_dataset("csv", data_files="train.csv")
    eval_dataset = eval_dataset.map(
        lambda e: tokenizer(e["data_LossDescription"], padding=True, truncation=True, max_length=128), batched=True
    )
    eval_dataset.set_format(
        type="torch",
        columns=["input_ids", "token_type_ids", "attention_mask", "labels"],
    )

    def my_metrics(p: EvalPrediction):
        # preds = np.squeeze(p.predictions)
        return {}

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset.get('train'),
        eval_dataset=eval_dataset.get('train'),
        tokenizer=tokenizer,
        compute_metrics=my_metrics,   # build_compute_metrics_fn(data_args.task_name),
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    eval_results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        eval_datasets = [eval_dataset]
        # if data_args.task_name == "mnli":
        #     mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
        #     eval_datasets.append(
        #         GlueDataset(mnli_mm_data_args, tokenizer=tokenizer, mode="dev", cache_dir=model_args.cache_dir)
        #     )

        for eval_dataset in eval_datasets:
            # trainer.compute_metrics = build_compute_metrics_fn(eval_dataset.args.task_name)
            eval_result = trainer.predict(test_dataset=eval_dataset.get('train'))
            print(eval_result)
            # eval_result = trainer.evaluate(eval_dataset=eval_dataset.get('train'))

            # output_eval_file = os.path.join(
            #     training_args.output_dir, f"eval_results_{eval_dataset.args.task_name}.txt"
            # )
            # if trainer.is_world_master():
            #     with open(output_eval_file, "w") as writer:
            #         logger.info("***** Eval results {} *****".format(eval_dataset.args.task_name))
            #         for key, value in eval_result.items():
            #             logger.info("  %s = %s", key, value)
            #             writer.write("%s = %s\n" % (key, value))

            # eval_results.update(eval_result)

    # if training_args.do_predict:
    #     logging.info("*** Test ***")
    #     test_datasets = [test_dataset]
    #     if data_args.task_name == "mnli":
    #         mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
    #         test_datasets.append(
    #             GlueDataset(mnli_mm_data_args, tokenizer=tokenizer, mode="test", cache_dir=model_args.cache_dir)
    #         )

    #     for test_dataset in test_datasets:
    #         predictions = trainer.predict(test_dataset=test_dataset).predictions
    #         if output_mode == "classification":
    #             predictions = np.argmax(predictions, axis=1)

    #         output_test_file = os.path.join(
    #             training_args.output_dir, f"test_results_{test_dataset.args.task_name}.txt"
    #         )
    #         if trainer.is_world_master():
    #             with open(output_test_file, "w") as writer:
    #                 logger.info("***** Test results {} *****".format(test_dataset.args.task_name))
    #                 writer.write("index\tprediction\n")
    #                 for index, item in enumerate(predictions):
    #                     if output_mode == "regression":
    #                         writer.write("%d\t%3.3f\n" % (index, item))
    #                     else:
    #                         item = test_dataset.get_labels()[item]
    #                         writer.write("%d\t%s\n" % (index, item))
    # return eval_results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()