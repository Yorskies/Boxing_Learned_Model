import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        # Layer linear untuk mempelajari bobot setiap timestep
        self.attention = nn.Linear(hidden_size * 2, 1, bias=False)

    def forward(self, lstm_outputs):
        # lstm_outputs shape: (batch_size, seq_length, hidden_size * 2)
        
        # 1. Hitung skor attention untuk tiap frame
        attn_scores = self.attention(lstm_outputs) # (batch, seq, 1)
        attn_scores = torch.squeeze(attn_scores, -1) # (batch, seq)
        
        # 2. Terapkan softmax agar total bobot seluruh frame = 1.0
        attn_weights = F.softmax(attn_scores, dim=1) # (batch, seq)
        
        # 3. Kalikan output LSTM dengan bobotnya
        attn_weights_expanded = attn_weights.unsqueeze(-1) # (batch, seq, 1)
        context_vector = torch.sum(lstm_outputs * attn_weights_expanded, dim=1) # (batch, hidden_size * 2)
        
        return context_vector, attn_weights

class AttentionBiLSTM(nn.Module):
    def __init__(self, input_size=132, hidden_size=64, num_layers=1, num_classes=4, dropout_rate=0.3):
        """
        Model klasifikasi berbasis Bidirectional LSTM dipadukan dengan Attention Mechanism.
        Didesain secara khusus agar parameternya mudah diubah mengikuti Skenario 1, 2, atau 3.
        """
        super(AttentionBiLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # PyTorch memberikan warning jika dropout di-set tapi num_layers=1.
        # Jadi kita paskan dropout LSTM jika num_layers > 1
        lstm_dropout = dropout_rate if num_layers > 1 else 0.0
        
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True,
            dropout=lstm_dropout
        )
        
        # Layer Attention
        self.attention = Attention(hidden_size)
        
        # Layer Dropout untuk mencegah overfitting sebelum masuk Linear
        self.dropout = nn.Dropout(dropout_rate)
        
        # Fully Connected Layer (Output)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        
        # 1. BiLSTM
        lstm_out, _ = self.lstm(x) # lstm_out shape: (batch, seq, hidden_size * 2)
        
        # 2. Attention
        context_vector, _ = self.attention(lstm_out) # context_vector shape: (batch, hidden_size * 2)
        
        # 3. Dropout
        out = self.dropout(context_vector)
        
        # 4. Classification
        out = self.fc(out) # (batch, num_classes)
        
        # Return raw logits karena nn.CrossEntropyLoss mengaplikasikan softmax secara internal
        return out
