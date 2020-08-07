# Change parameters of model
target_update_interval = 1000
alpha = .1
experience_buffer_size = 2000
experience_sample_size = 15
gamma = .1
history_length = 10
epsilon_min = .1
epsilon_max = 1.0
epsilon_dec_steps = 5
epsilon_dec = (epsilon_max - epsilon_min) / epsilon_dec_steps
max_steps = 40

training = False

# training_ratio = 0.8