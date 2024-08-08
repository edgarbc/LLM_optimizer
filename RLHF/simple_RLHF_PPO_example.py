# Adapted from Introduction to LLMs in Python by DataCamp
# Aug 2024

from trl import PPOTrainer, PPOConfig, create_reference_model, AutoModelForCausalLMWithValueHead
from trl.core import respond_to_batch

model = AutoModelForCausalLMWithValueHead.from_pretrained('sshleifer/tiny-gpt2') # or 'gpt2'

# Instantiate a reference model
model_ref = create_reference_model(model)

tokenizer = AutoTokenizer.from_pretrained('sshleifer/tiny-gpt2')

if tokenizer._pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Initialize trainer configuration
ppo_config = PPOConfig(batch_size=1, mini_batch_size=1)

prompt = "Next year, I "
input = tokenizer.encode(prompt, return_tensors="pt")

response  = respond_to_batch(model, input)

# Create a PPOTrainer instance
ppo_trainer = PPOTrainer(ppo_config, model, model_ref, tokenizer)

# for now we will add reward but this are the values coming from human feedback
reward = [torch.tensor(1.0)]

# Train LLM for one step with PPO
train_stats = ppo_trainer.step([input[0]], [response[0]], reward)
