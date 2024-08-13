import torch
import torch.nn as nn
from torch.nn import functional as F
import lightning as L

class LSTMClassifier(L.LightningModule):
    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, device):
        super().__init__()
        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length
        self.d = device
        
        self.word_embeddings = nn.Embedding(vocab_size, embedding_length)#
        self.lstm = nn.LSTM(embedding_length, hidden_size)
        self.label = nn.Linear(hidden_size, output_size)

        self.validation_step_outputs = []

   
    def forward(self, input_sentence):
        input = self.word_embeddings(input_sentence) # embedded input of shape = (batch_size, num_sequences,  embedding_length)
        input = input.permute(1, 0, 2) # input.size() = (num_sequences, batch_size, embedding_length)
        h_0 = torch.zeros(1, self.batch_size, self.hidden_size).to(self.d) # Initial hidden state of the LSTM
        c_0 = torch.zeros(1, self.batch_size, self.hidden_size).to(self.d) # Initial cell state of the LSTM
        output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))
        final_output = self.label(final_hidden_state[-1]) # final_hidden_state.size() = (1, batch_size, hidden_size) & final_output.size() = (batch_size, output_size)
        
        return final_output


    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x = batch['input_ids']
        score = self.forward(x)

        loss = F.mse_loss(score, batch['labels'])
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['input_ids']
        y = batch['labels']
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("valid_loss", loss, on_epoch=True)
        self.validation_step_outputs.append(loss.item())

    def on_validation_epoch_end(self):
        print(f"Validation loss: {sum(self.validation_step_outputs)/len(self.validation_step_outputs)}")
        self.validation_step_outputs.clear()  # free memory

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

