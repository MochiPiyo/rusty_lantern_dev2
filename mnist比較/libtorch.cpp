#include <torch/torch.h>
#include <iostream>

// ニューラルネットワークの定義
struct Net : torch::nn::Module {
    Net() {
        fc1 = register_module("fc1", torch::nn::Linear(28 * 28, 128));
        fc2 = register_module("fc2", torch::nn::Linear(128, 64));
        fc3 = register_module("fc3", torch::nn::Linear(64, 10));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = x.view({-1, 28 * 28});
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        x = fc3->forward(x);
        return x;
    }

    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};

int main() {
    // データセットの設定
    auto train_dataset = torch::data::datasets::MNIST("./data")
                             .map(torch::data::transforms::Normalize<>(0.5, 0.5))
                             .map(torch::data::transforms::Stack<>());
    auto train_loader = torch::data::make_data_loader(std::move(train_dataset), 64);

    // モデルとオプティマイザの設定
    Net model;
    torch::optim::SGD optimizer(model.parameters(), 0.01);

    // 訓練ループ
    for (size_t epoch = 0; epoch < 5; ++epoch) {
        for (auto& batch : *train_loader) {
            auto data = batch.data.view({-1, 28 * 28});
            auto targets = batch.target;

            // 勾配をゼロに初期化
            optimizer.zero_grad();
            // 順伝播 + 誤差逆伝播 + 重み更新
            auto output = model.forward(data);
            auto loss = torch::nn::functional::cross_entropy(output, targets);
            loss.backward();
            optimizer.step();
        }
        std::cout << "Epoch: " << epoch + 1 << " Loss: " << loss.item<float>() << std::endl;
    }

    return 0;
}
