import os
from Sequence_labeling.utils import *
from Sequence_labeling.model import *


dir_path = '' #set the root path

train_file = os.path.join(dir_path, 'kg_data/train.txt')  # path to training data
val_file = os.path.join(dir_path, 'kg_data/val.txt')  # path to validation data

word_file = os.path.join(dir_path, 'kg_data/embedding/word.pkl')  # path to pre-trained word embeddings
emb_file = os.path.join(dir_path, 'kg_data/embedding/word_embedding.txt')
bios_label_file = os.path.join(dir_path, 'kg_answer/kg_data/bios_label.txt')
path_save_model = os.path.join(dir_path, 'kg_answer/kg_data/seq_label.model')
max_sequence_size = 50
emb_dim = 100
HIDDEN_DIM = 100
epochs = 100
START_TAG = "<START>"
STOP_TAG = "<STOP>"
check_per_step = 50

def main():
    word_map, emb_dict = create_embedding(word_path=word_file, emb_path=emb_file, emb_size=emb_dim)

    tags_map = get_tag_map(bios_label_file, START_TAG, STOP_TAG)


    train_words, train_tags = corpus_process(train_file, tags_map, word_map)

    val_words, val_tags = corpus_process(val_file, tags_map, word_map)

    train_inputs, train_tags = create_input_tensor(train_words, train_tags, word_map, max_sequence_size)
    val_inputs, val_tags = create_input_tensor(train_words, train_tags, word_map, max_sequence_size)

    model = BiLSTM_CRF(len(word_map), tags_map, emb_dim, HIDDEN_DIM, START_TAG, STOP_TAG)
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

    # Make sure prepare_sequence from earlier in the LSTM section is loaded
    min_loss = 10000
    for epoch in range(epochs):  # again, normally you would NOT do 300 epochs, it is toy data

        step = 0
        loss = 0
        for input, tags in zip(train_inputs, train_tags):
            step +=1
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Get our inputs ready for the network, that is,
            # turn them into Variables of word indices.
            # sentence_in = prepare_sequence(input, word_to_ix)
            # print(tags)
            input_ = torch.LongTensor(input)
            targets = torch.LongTensor(tags)

            # Step 3. Run our forward pass.
            neg_log_likelihood = model.neg_log_likelihood(input_, targets)

            loss += neg_log_likelihood[0]
            # Step 4. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            neg_log_likelihood.backward()
            optimizer.step()

            if step % check_per_step == 0:
                print('epoch: {}, step: {}, loss: {}'.format(epoch, step, neg_log_likelihood[0]))

        one_epoch_loss = loss/len(train_inputs)
        if one_epoch_loss < min_loss:
            min_loss = one_epoch_loss
            torch.save(model.state_dict(), path_save_model)
            print('Save Model!   epoch: {}, loss: {}'.format(epoch, min_loss))

        print(epoch, loss/len(train_inputs))
if __name__ == '__main__':
    main()