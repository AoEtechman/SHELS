import torch
import torch.nn as nn
import torch.nn.functional as F


class AUDIO_MNIST(nn.Module):
    def __init__(self, output_dim = 9, cosine_sim = True, baseline = False):
        super().__init__()
        self.features = nn.Sequential(
             nn.Conv2d(in_channels = 1, out_channels = 96, kernel_size = 11, stride = 4, padding=(1, 1), bias = True),
             nn.ReLU(inplace = True),
             nn.Conv2d(in_channels = 96, out_channels = 96, kernel_size = 3, padding=(1, 1), bias = True),
             nn.ReLU(inplace = True),
             nn.MaxPool2d(kernel_size = 3, stride = 2),
             #nn.Dropout(0.25),
             nn.Conv2d(in_channels = 96, out_channels = 192, kernel_size = 5,  padding=(1, 1), bias = True),
             nn.ReLU(inplace = True),
             nn.Conv2d(in_channels = 192, out_channels = 192, kernel_size = 5,  padding=(1, 1), bias = True),
             nn.ReLU(inplace = True),
             nn.MaxPool2d(kernel_size = 3, stride = 2),
             #nn.Dropout(0.25),
             nn.Conv2d(in_channels = 192, out_channels = 384, kernel_size = 3,  padding=(1, 1), bias = True),
             nn.ReLU(inplace = True),
             nn.Conv2d(in_channels = 384, out_channels = 384, kernel_size = 3,  padding=(1, 1), bias = True),
             nn.ReLU(inplace = True),
             nn.MaxPool2d(kernel_size = 3, stride = 2),
             nn.Flatten()
        )

        self.classifier = nn.Sequential(
             nn.Linear(9600,256, bias = True),     ## 17
             #nn.ReLU(inplace = True),
             #nn.Linear(512,128, bias = True),          ## 19
             nn.ReLU(inplace = True),
             nn.Linear(256, output_dim, bias = True),  ## 21
             #nn.ReLU(inplace = True),
             #nn.Linear(32, output_dim) ## 23
        )
        if cosine_sim:
            self.fc_scale = nn.Linear(256,1)
            self.bn_scale = nn.BatchNorm1d(1)

        self.baseline = baseline
        self.cosine_sim = cosine_sim

    def forward(self, x):
        layer_output = []
        # t = x
        x = x.type(torch.float32)
        
        feats = x
        
        for layer in self.features:

            feats = layer(feats)
            layer_output.append(feats)

        out = feats
        # print(out.shape)
        for layer in self.classifier:
            out = layer(out)
            layer_output.append(out)
        if not self.cosine_sim:
            return layer_output, out
        ## cosine
        f = layer_output[len(layer_output)-2]
        scale = torch.exp(self.bn_scale(self.fc_scale(f))) #puts last features through the final layers and then batch norm 1d
        weight = layer.weight
        f_norm = F.normalize(f)
        weight_norm = F.normalize(weight)
        weight_norm_transposed = torch.transpose(weight_norm, 0, 1)
        out = torch.mm(f_norm,weight_norm_transposed) # the denominator
        scaled_output = scale*out # multiply 

        layer_output[len(layer_output)-1] = out # last layer output is now the multiplication of input and weight norms
        softmax = F.softmax(scaled_output, 1) # softmax of scale and output from old output from linear layer
        relu = F.relu(out) 
        if self.baseline:
            return layer_output, softmax
        else:
            return layer_output, out



    def freeze_conv_weights(self):

        for i in range(0, len(self.features)):
            self.features[i].requires_grad_(False)

        self.classifier[0].requires_grad_(False)
        # self.classifier[2].requires_grad_(False)
        # self.classifier[4].requires_grad_(False)


    def update_model_weights(self, weights_f,weights,nodes_f, nodes):
        with torch.no_grad():
            idx = 0
            for i in range(0,len(self.features)):
                if isinstance(self.features[i], nn.Conv2d):
                    self.features[i].weight.data = weights_f[idx]
                    # self.features[i].bias.data = self.features[i].bias.data.clone().detach()*torch.Tensor(nodes_f[idx]).to("cuda:0")
                    idx+=1
            idx = 0
            for i in range(0,len(self.classifier)):
                if isinstance(self.classifier[i], nn.Linear):
                    self.classifier[i].weight.data = weights[idx]

                    # self.classifier[i].bias.data = self.classifier[i].bias.data.clone().detach()*torch.Tensor(nodes[idx]).to("cuda:0")
                    idx+=1


    def increment_classes(self, n):

        in_features = self.classifier[2].in_features
        out_features = self.classifier[2].out_features
        weight = self.classifier[2].weight.data
        bias = self.classifier[2].bias.data

        self.classifier[2] = nn.Linear(in_features, out_features+n, bias=True)

        self.classifier[2].weight.data[:out_features] = weight

        m = torch.zeros_like(self.classifier[2].weight.data[out_features:])

        t= torch.nn.init.xavier_uniform_(m, gain=1.0)
        self.classifier[2].weight.data[out_features:] = t[0]

        nn.init.constant_(self.classifier[2].bias,0)
        self.classifier[2].to('cuda:0')

