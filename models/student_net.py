import torch


class build_student_net(torch.nn.Module):
    def __init__(self, in_channels, stages=3, student_dim=100):
        super(build_student_net, self).__init__()
        self.student = Student(in_channels, stages, student_dim)

    def forward(self, x, mask=None):
        if mask is not None:
            masked_x = x * (1 - mask)
        else:
            masked_x = x
        s_out_list = self.student(masked_x)

        return s_out_list


class Student(torch.nn.Module):
    def __init__(self, in_channels, stages=3, hidden_dim=100):
        super(Student, self).__init__()
        self.layers = torch.nn.ModuleList([])
        for _ in range(stages - 1):
            self.layers.append(torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, hidden_dim, 3, 1, 1),
                torch.nn.BatchNorm2d(hidden_dim),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose2d(hidden_dim, in_channels, 3, 1, 1),
                torch.nn.BatchNorm2d(in_channels),
                torch.nn.ReLU(),
            ))
        self.out_layer = torch.nn.Linear(in_channels, in_channels)

    def forward(self, x):
        # x: [B, C, M, N]
        out_list = []
        for layer in self.layers:
            x = layer(x)
            out_list.append(x)
        x = torch.permute(x, [0, 2, 3, 1])
        B, M, N, C = x.shape
        x = torch.reshape(x, [B, M * N, C])
        out_list.append(self.out_layer(x).reshape([B, M, N, C]).permute([0, 3, 1, 2]))

        return out_list
