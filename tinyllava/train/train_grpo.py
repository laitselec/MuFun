# train_grpo.py
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from tinyllava.model.modeling_tinyllava import TinyLlavaForConditionalGeneration
from tinyllava.model.configuration_tinyllava import TinyLlavaConfig
from packaging import version
import pathlib
import math
import tokenizers
import transformers


from tinyllava.train.tinyllava_trainer import LLaVATrainer
from tinyllava.training_recipe import TrainingRecipeFactory
from tinyllava.utils import *
from tinyllava.model import *
from tinyllava.data.dataset_rl import make_rl_data_module

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


def load_settings(model_arguments, data_arguments, training_arguments):
    model_arguments.tune_type_connector = training_arguments.tune_type_connector
    model_arguments.tune_type_llm = training_arguments.tune_type_llm
    model_arguments.tune_type_vision_tower = training_arguments.tune_type_vision_tower
    model_arguments.image_aspect_ratio = data_arguments.image_aspect_ratio

    model_args = {}
    model_args['llm'] = _load_llm_settings(model_arguments)
    model_args['vision_tower'] = _load_vision_settings(model_arguments)
    model_args['connector'] = _load_connector_settings(model_arguments) 
    return model_args

def _load_llm_settings(model_arguments):
    llm_args = {}
    llm_args['model_name_or_path'] = model_arguments.model_name_or_path
    llm_args['cache_dir'] = model_arguments.cache_dir
    llm_args['attn_implementation'] = model_arguments.attn_implementation # flash_attention_2 only supports torch.float16 and torch.bfloat16 dtypes
    return llm_args

def _load_vision_settings(model_arguments):
    vision_args = {}
    vision_args['model_name_or_path'] = model_arguments.vision_tower.split(':')[-1]
    if model_arguments.vision_tower2 != '':
        vision_args['model_name_or_path2'] = model_arguments.vision_tower2.split(':')[-1]
    return vision_args

def _load_connector_settings(model_arguments):
    connector_args = {}
    connector_args['connector_type'] = model_arguments.connector_type
    return connector_args


def train():
    
    # load argument
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_arguments, data_arguments, training_arguments = parser.parse_args_into_dataclasses()
    
    logger_setting(getattr(training_arguments, 'output_dir', None))

    training_recipe = TrainingRecipeFactory(training_arguments.training_recipe)(training_arguments) 
    # model_args contain arguements for huggingface model .from_pretrained function
    model_args = load_settings(model_arguments, data_arguments, training_arguments)
    model_args = training_recipe.add_args(model_args)
    model_config = TinyLlavaConfig()
    model_config.load_from_config(model_arguments)
    model = TinyLlavaForConditionalGeneration(model_config)
    # load pretrained checkpoint
    if training_arguments.pretrained_model_path is not None:
        model = training_recipe.load(model, model_args)
    else:
        model.load_llm(**model_args['llm'])
        model.load_vision_tower(**model_args['vision_tower'])
        model.load_connector(**model_args['connector'])

    model = training_recipe(model)
    model.config.use_cache = False
    model.config.image_aspect_ratio = data_arguments.image_aspect_ratio
    tokenizer = model.tokenizer
    data_arguments.image_processor = model.vision_tower._image_processor
    data_arguments.is_multimodal = True
    data_module = make_rl_data_module(tokenizer=tokenizer,
                                              data_args=data_arguments)
    
   
    
    def reward_len(completions, **kwargs):
        # print(f"Completions: {completions}")
        return [-abs(20 - len(completion)) for completion in completions]

    # choices = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    def gt_single(completion, label):
        reward = 0.0
        cnt = completion.count(label)
        if cnt==1:
            reward = 1.0
        # if label in completion:
        #     reward = 1.0
            # for choice in choices:
            #     if choice in completion and choice != label:
            #         reward -= 0.1
        # if len(completion) > 16:
        #     reward -= 0.5*math.exp((len(completion)-32)/16.0)
        return reward
    
    def reward_gt(completions, gt, **kwargs):
        print(f"Completions: {completions}")
        
        return [gt_single(completion, label.split('. ')[-1]) for completion, label in zip(completions, gt)]
    
    def calculate_wer(reference, hypothesis):
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        # Counting the number of substitutions, deletions, and insertions
        substitutions = sum(1 for ref, hyp in zip(ref_words, hyp_words) if ref != hyp)
        deletions = len(ref_words) - len(hyp_words)
        insertions = len(hyp_words) - len(ref_words)
        # Total number of words in the reference text
        total_words = len(ref_words)
        # Calculating the Word Error Rate (WER)
        wer = (substitutions + deletions + insertions) / total_words
        return wer
    
    def reward_wer(completions, gt, **kwargs):
        print(f"Completions: {completions}")

        return [1.0 - calculate_wer(label, completion) for completion, label in zip(completions, gt)]

    training_args = GRPOConfig()
    for k, v in training_arguments.__dict__.items():
        if k in training_args.__dict__:
            training_args.__dict__[k] = v
    # num_processes = 5
    # # training_args.loss_type = 'grpo'
    # training_args.num_generations = 5
    # training_args.generation_batch_size = training_args.per_device_train_batch_size * num_processes * training_args.gradient_accumulation_steps
    # training_args.steps_per_generations = training_args.gradient_accumulation_steps
    # training_args.max_completion_length = 256
    # training_args.temperature = 0.8
    
    # training_args.beta = 0.005
    ref_model = None
    if training_args.beta != 0.0:
        ref_model_name = training_arguments.pretrained_model_path
        ref_model = TinyLlavaForConditionalGeneration.from_pretrained(ref_model_name,low_cpu_mem_usage=True)
    
    # training_args.ds3_gather_for_generation = False
    # print(f"Training arguments: {training_args}")
    
    trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_gt,
    args=training_args,
    **data_module,
    ref_model=ref_model,
    # train_dataset=dataset,
)
    
    trainer.train()
    
    training_recipe.save(model, trainer)

if __name__ == "__main__":
    train()
    
