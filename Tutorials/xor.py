from __future__ import print_function
from __future__ import division
import cntk
import cntk.ops
import cntk.layers
import cntk.io
import cntk.train
from cntk.initializer import *

# Data preparation
INPUT_DATA = r'''|xy 0 0	|r 0
|xy 1 0	|r 1
|xy 0 1	|r 1
|xy 1 1	|r 0
'''

input_file = 'input'
with open(input_file, 'w') as f:
    f.write(INPUT_DATA)

# Network creation
xy = cntk.input_variable(2)
label = cntk.input_variable(1)

model = cntk.layers.Sequential([
    cntk.layers.Dense(2, activation=cntk.ops.tanh),
    cntk.layers.Dense(1)])

z = model(xy)
loss = cntk.squared_error(z, label)

# Learner and trainer
learner = cntk.sgd(model.parameters, cntk.learning_rate_schedule(0.1, cntk.UnitType.sample))
progress_writer = cntk.logging.ProgressPrinter(freq=1)
trainer = cntk.train.Trainer(z, (loss, loss), learner, [progress_writer])

# Minibatch source creation
mb_source = cntk.io.MinibatchSource(cntk.io.CTFDeserializer(input_file, cntk.io.StreamDefs(
    xy=cntk.io.StreamDef(field='xy', shape=2),
    r=cntk.io.StreamDef(field='r', shape=1)
)))

# Run a manual training minibatch loop
minibatch_size = 4
input_map = { xy : mb_source['xy'], label : mb_source['r'] }
max_samples = 800
train = True
while train and trainer.total_number_of_samples_seen < max_samples:
    data = mb_source.next_minibatch(minibatch_size, input_map)
    train = trainer.train_minibatch(data)

# Run a manual evaluation loop
test_mb_source = cntk.io.MinibatchSource(cntk.io.CTFDeserializer(input_file, cntk.io.StreamDefs(
    xy=cntk.io.StreamDef(field='xy', shape=2),
    r=cntk.io.StreamDef(field='r', shape=1)
)), randomize=False, max_samples = 1000)

input_map = { xy : test_mb_source['xy'], label : test_mb_source['r'] }
total_samples = 0
error = 0.
while True:
    data = test_mb_source.next_minibatch(32, input_map)
    if not data:
        break
    total_samples += data[label].number_of_samples 
    error += trainer.test_minibatch(data) * data[label].number_of_samples

print("Error %f" % (error / total_samples))


# Run a manual training minibatch loop

# Trying to restore
mb_source = cntk.io.MinibatchSource(cntk.io.CTFDeserializer(input_file, cntk.io.StreamDefs(
    xy=cntk.io.StreamDef(field='xy', shape=2),
    r=cntk.io.StreamDef(field='r', shape=1)
)))
learner = cntk.sgd(model.parameters, cntk.learning_rate_schedule(0.1, cntk.UnitType.sample))
progress_writer = cntk.logging.ProgressPrinter(freq=1)
trainer = cntk.train.Trainer(z, (loss, loss), learner, [progress_writer])

mb_source_state = trainer.restore_from_checkpoint("test")
mb_source.restore_from_checkpoint(mb_source_state)

minibatch_size = 4
input_map = { xy : mb_source['xy'], label : mb_source['r'] }
max_samples = 800
samples_seen = 0
checkpoint_frequency = 100
last_checkpoint = -1
train = True
while train and trainer.total_number_of_samples_seen < max_samples:
    data = mb_source.next_minibatch(minibatch_size, input_map)
    train = trainer.train_minibatch(data)
    if trainer.total_number_of_samples_seen / checkpoint_frequency != last_checkpoint:
        mb_source_state = mb_source.get_checkpoint_state()
        trainer.save_checkpoint(mb_source_state, "test")
        last_checkpoint = trainer.total_number_of_samples_seen / checkpoint_frequency


# Run a manual training minibatch loop

mb_source = cntk.io.MinibatchSource(cntk.io.CTFDeserializer(input_file, cntk.io.StreamDefs(
    xy=cntk.io.StreamDef(field='xy', shape=2),
    r=cntk.io.StreamDef(field='r', shape=1)
)))

# Make sure the learner is distributed
learner = cntk.distributed.data_parallel_learner(cntk.sgd(model.parameters, cntk.learning_rate_schedule(0.1, cntk.UnitType.sample)))
progress_writer = cntk.logging.ProgressPrinter(freq=1)
trainer = cntk.train.Trainer(z, (loss, loss), learner, [progress_writer])

mb_source_state = trainer.restore_from_checkpoint("test")
mb_source.restore_from_checkpoint(mb_source_state)

minibatch_size = 4
input_map = { xy : mb_source['xy'], label : mb_source['r'] }
max_samples = 800
samples_seen = 0
checkpoint_frequency = 100
last_checkpoint = -1
train = True
while train and trainer.total_number_of_samples_seen < max_samples:
    # Make sure each worker gets its own data only
    data = mb_source.next_minibatch(minibatch_size, input_map, distributed.Communicator.rank, distributed.Communicator.num_workers)
    train = trainer.train_minibatch(data)
    if trainer.total_number_of_samples_seen / checkpoint_frequency != last_checkpoint:
        mb_source_state = mb_source.get_checkpoint_state()
        trainer.save_checkpoint(mb_source_state, "test")
        last_checkpoint = trainer.total_number_of_samples_seen / checkpoint_frequency

# Finalize MPI
distributed.Communicator.finalize()

