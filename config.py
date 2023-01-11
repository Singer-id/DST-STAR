# Copyright (c) Facebook, Inc. and its affiliates

import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_checkpoint", type=str, default="t5-small", help="Path, url or short name of the model")
    parser.add_argument("--saving_dir", type=str, default="save", help="Path for saving")
    parser.add_argument("--train_batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--meta_batch_size", type=int, default=1, help="Batch size for meta training")
    parser.add_argument("--dev_batch_size", type=int, default=128, help="Batch size for validation")
    parser.add_argument("--test_batch_size", type=int, default=1, help="Batch size for test")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=557, help="Random seed")
    parser.add_argument("--verbose", action='store_true', help="continual baseline")
    parser.add_argument("--length", type=int, default=50, help="Batch size for validation")
    parser.add_argument("--max_history", type=int, default=2, help="max number of turns in the dialogue")
    parser.add_argument("--GPU", type=int, default=1, help="number of gpu to use")
    parser.add_argument("--pre_model_name", type=str, default="fine_tune_bert")
    parser.add_argument("--model_name", type=str, default="DST_model")
    parser.add_argument("--slot_lang", type=str, default="slottype", help="use 'none', 'human', 'naive', 'value', 'question', 'slottype' slot description")
    parser.add_argument("--fewshot", type=float, default=0.0, help="data ratio for few shot experiment")
    parser.add_argument("--fix_label", action='store_true')
    parser.add_argument("--except_domain", type=str, default="none", help="hotel, train, restaurant, attraction, taxi")
    parser.add_argument("--only_domain", type=str, default="none", help="hotel, train, restaurant, attraction, taxi")
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--semi", action='store_true')
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--auxiliary_task_ratio", type=float, default=1.0, help="auxiliary task data amount / main task data amount, use 1.0, 0.5, 0.375, 0.25, 0.125")
    parser.add_argument("--base", action='store_true')
    parser.add_argument("--task2first", action='store_true')
    parser.add_argument("--use_2.4", action='store_true')
    parser.add_argument("--data_process", action='store_true')
    parser.add_argument("--data_loader", action='store_true')
    parser.add_argument("--pre_train_data", action='store_true')
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--continue_train", action='store_true')
    parser.add_argument("--crf", action='store_true')
    parser.add_argument("--contrastive_size", type=int, default=16, help="Batch size for contrastive_size")
    # star
    parser.add_argument("--data_dir", default='data/mwz2.1', type=str)
    parser.add_argument("--pretrained_model", default='bert-base-uncased', type=str)
    parser.add_argument("--save_dir", default='out-bert/exp', type=str)
    parser.add_argument("--attn_type", default='softmax', type=str,
                        help="softmax or tanh")

    parser.add_argument("--random_seed", default=42, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    # parser.add_argument("--train_batch_size", default=16, type=int)
    parser.add_argument("--enc_warmup", default=0.1, type=float)
    parser.add_argument("--dec_warmup", default=0.1, type=float)
    parser.add_argument("--enc_lr", default=4e-5, type=float)
    parser.add_argument("--dec_lr", default=1e-4, type=float)

    # parser.add_argument("--n_epochs", default=12, type=int)
    parser.add_argument("--eval_epoch", default=1, type=int)
    parser.add_argument("--eval_step", default=10000, type=int,
                        help="Within each epoch, do evaluation as well at every eval_step")

    parser.add_argument("--dropout_prob", default=0.1, type=float)
    parser.add_argument("--word_dropout", default=0.1, type=float)

    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--patience", default=8, type=int)
    parser.add_argument("--attn_head", default=4, type=int)
    parser.add_argument("--num_history", default=20, type=int)
    parser.add_argument("--distance_metric", default="euclidean", type=str,
                        help="euclidean or cosine")

    parser.add_argument("--num_self_attention_layer", default=6, type=int)
    args = parser.parse_args()
    return args
