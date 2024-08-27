#ifndef READERMNIST_H
#define READERMNIST_H

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cstdint>
#include <memory>
#include "container.h"

//class readerMNIST
//{
//public:
//    readerMNIST();
//};


namespace mnistFachon {

uint32_t read_header(const std::unique_ptr<char[]>& buffer, size_t position);

std::unique_ptr<char[]> read_mnist_file(const std::string& path, uint32_t key);

//////////////////////////////////////////////////////////


template <typename Pixel = uint8_t, typename Label = uint8_t>
struct MNIST_dataset {
    std::vector<std::vector<Pixel>> training_images; ///< The training images
    std::vector<std::vector<Pixel>> test_images;     ///< The test images
    std::vector<Label> training_labels;              ///< The training labels
    std::vector<Label> test_labels;                  ///< The test labels
};


template <typename Pixel = uint8_t, typename Label = uint8_t>
std::vector<std::vector<Pixel>> read_mnist_image_file(const std::string& path) {
    auto buffer = read_mnist_file(path, 0x803);

    if (buffer) {
        auto count   = read_header(buffer, 1);
        auto rows    = read_header(buffer, 2);
        auto columns = read_header(buffer, 3);

        //Skip the header
        //Cast to unsigned char is necessary cause signedness of char is
        //platform-specific
        auto image_buffer = reinterpret_cast<unsigned char*>(buffer.get() + 16);

        std::vector<std::vector<Pixel>> images;
        images.reserve(count);

        for (size_t i = 0; i < count; ++i) {
            images.emplace_back(rows * columns);

            for (size_t j = 0; j < rows * columns; ++j) {
                auto pixel   = *image_buffer++;
                images[i][j] = static_cast<Pixel>(pixel);
            }
        }

        return images;
    }

    return {};
}

template <typename Label = uint8_t>
std::vector<Label> read_mnist_label_file(const std::string& path) {
    auto buffer = read_mnist_file(path, 0x801);

    if (buffer) {
        auto count = read_header(buffer, 1);

        auto label_buffer = reinterpret_cast<unsigned char*>(buffer.get() + 8);

        std::vector<Label> labels(count);

        for (size_t i = 0; i < count; ++i) {
            auto label = *label_buffer++;
            labels[i]  = static_cast<Label>(label);
        }

        return labels;
    }

    return {};
}

template <typename Pixel = uint8_t>
std::vector<std::vector<Pixel>> read_training_images() {
    return read_mnist_image_file<Pixel>("mnist/train-images-idx3-ubyte");
}


template <typename Pixel = uint8_t>
std::vector<std::vector<Pixel>> read_test_images() {
    return read_mnist_image_file<Pixel>("mnist/t10k-images-idx3-ubyte");
}

template <typename Label = uint8_t>
std::vector<Label> read_training_labels() {
    return read_mnist_label_file<Label>("mnist/train-labels-idx1-ubyte");
}

template <typename Label = uint8_t>
std::vector<Label> read_test_labels() {
    return read_mnist_label_file<Label>("mnist/t10k-labels-idx1-ubyte");
}

template <typename Pixel = uint8_t, typename Label = uint8_t>
MNIST_dataset<Pixel, Label> read_dataset() {
    MNIST_dataset<Pixel, Label> dataset;

    dataset.training_images = read_training_images<Pixel>();
    dataset.training_labels = read_training_labels<Label>();

    dataset.test_images = read_test_images<Pixel>();
    dataset.test_labels = read_test_labels<Label>();

    return dataset;
}




template <template <typename...> class Container, typename Image, typename Label>
struct MNIST_dataset {
    Container<Image> training_images; ///< The training images
    Container<Image> test_images;     ///< The test images
    Container<Label> training_labels; ///< The training labels
    Container<Label> test_labels;     ///< The test labels


    void resize_training(std::size_t new_size) {
        if (training_images.size() > new_size) {
            training_images.resize(new_size);
            training_labels.resize(new_size);
        }
    }


    void resize_test(std::size_t new_size) {
        if (test_images.size() > new_size) {
            test_images.resize(new_size);
            test_labels.resize(new_size);
        }
    }
};

template <typename Container>
bool read_mnist_image_file_flat(Container& images, const std::string& path, std::size_t limit, std::size_t start = 0) {
    auto buffer = read_mnist_file(path, 0x803);

    if (buffer) {
        auto count   = read_header(buffer, 1);
        auto rows    = read_header(buffer, 2);
        auto columns = read_header(buffer, 3);

        //Skip the header
        //Cast to unsigned char is necessary cause signedness of char is
        //platform-specific
        auto image_buffer = reinterpret_cast<unsigned char*>(buffer.get() + 16);

        if (limit > 0 && count > limit) {
            count = static_cast<unsigned int>(limit);
        }

        // Ignore "start" first elements
        image_buffer += start * (rows * columns);

        for (size_t i = 0; i < count; ++i) {
            for (size_t j = 0; j < rows * columns; ++j) {
                images(i)[j] = *image_buffer++;
            }
        }

        return true;
    } else {
        return false;
    }
}


template <template <typename...> class Container = std::vector, typename Image, typename Functor>
void read_mnist_image_file(Container<Image>& images, const std::string& path, std::size_t limit, Functor func) {
    auto buffer = read_mnist_file(path, 0x803);

    if (buffer) {
        auto count   = read_header(buffer, 1);
        auto rows    = read_header(buffer, 2);
        auto columns = read_header(buffer, 3);

        //Skip the header
        //Cast to unsigned char is necessary cause signedness of char is
        //platform-specific
        auto image_buffer = reinterpret_cast<unsigned char*>(buffer.get() + 16);

        if (limit > 0 && count > limit) {
            count = static_cast<unsigned int>(limit);
        }

        images.reserve(count);

        for (size_t i = 0; i < count; ++i) {
            images.push_back(func());

            for (size_t j = 0; j < rows * columns; ++j) {
                auto pixel   = *image_buffer++;
                images[i][j] = static_cast<typename Image::value_type>(pixel);
            }
        }
    }
}


template <template <typename...> class Container = std::vector, typename Label = uint8_t>
void read_mnist_label_file(Container<Label>& labels, const std::string& path, std::size_t limit = 0) {
    auto buffer = read_mnist_file(path, 0x801);

    if (buffer) {
        auto count = read_header(buffer, 1);

        //Skip the header
        //Cast to unsigned char is necessary cause signedness of char is
        //platform-specific
        auto label_buffer = reinterpret_cast<unsigned char*>(buffer.get() + 8);

        if (limit > 0 && count > limit) {
            count = static_cast<unsigned int>(limit);
        }

        labels.resize(count);

        for (size_t i = 0; i < count; ++i) {
            auto label = *label_buffer++;
            labels[i]  = static_cast<Label>(label);
        }
    }
}


template <typename Container>
bool read_mnist_label_file_flat(Container& labels, const std::string& path, std::size_t limit = 0) {
    auto buffer = read_mnist_file(path, 0x801);

    if (buffer) {
        auto count = read_header(buffer, 1);

        //Skip the header
        //Cast to unsigned char is necessary cause signedness of char is
        //platform-specific
        auto label_buffer = reinterpret_cast<unsigned char*>(buffer.get() + 8);

        if (limit > 0 && count > limit) {
            count = static_cast<unsigned int>(limit);
        }

        for (size_t i = 0; i < count; ++i) {
            labels(i)  = *label_buffer++;
        }

        return true;
    } else {
        return false;
    }
}


template <typename Container>
bool read_mnist_label_file_categorical(Container& labels, const std::string& path, std::size_t limit = 0, std::size_t start = 0) {
    auto buffer = read_mnist_file(path, 0x801);

    if (buffer) {
        auto count = read_header(buffer, 1);

        //Skip the header
        //Cast to unsigned char is necessary cause signedness of char is
        //platform-specific
        auto label_buffer = reinterpret_cast<unsigned char*>(buffer.get() + 8);

        if (limit > 0 && count > limit) {
            count = static_cast<unsigned int>(limit);
        }

        // Ignore "start" first elements
        label_buffer += start;

        for (size_t i = 0; i < count; ++i) {
            labels(i)(static_cast<size_t>(*label_buffer++)) = 1;
        }

        return true;
    } else {
        return false;
    }
}


template <template <typename...> class Container = std::vector, typename Image, typename Functor>
Container<Image> read_training_images(const std::string& folder, std::size_t limit, Functor func) {
    Container<Image> images;
    read_mnist_image_file<Container, Image>(images, folder + "/train-images-idx3-ubyte", limit, func);
    return images;
}


template <template <typename...> class Container = std::vector, typename Image, typename Functor>
Container<Image> read_test_images(const std::string& folder, std::size_t limit, Functor func) {
    Container<Image> images;
    read_mnist_image_file<Container, Image>(images, folder + "/t10k-images-idx3-ubyte", limit, func);
    return images;
}


template <template <typename...> class Container = std::vector, typename Label = uint8_t>
Container<Label> read_training_labels(const std::string& folder, std::size_t limit) {
    Container<Label> labels;
    read_mnist_label_file<Container, Label>(labels, folder + "/train-labels-idx1-ubyte", limit);
    return labels;
}


template <template <typename...> class Container = std::vector, typename Label = uint8_t>
Container<Label> read_test_labels(const std::string& folder, std::size_t limit) {
    Container<Label> labels;
    read_mnist_label_file<Container, Label>(labels, folder + "/t10k-labels-idx1-ubyte", limit);
    return labels;
}


template <template <typename...> class Container, typename Image, typename Label = uint8_t>
MNIST_dataset<Container, Image, Label> read_dataset_3d(const std::string& folder, std::size_t training_limit = 0, std::size_t test_limit = 0) {
    MNIST_dataset<Container, Image, Label> dataset;

    dataset.training_images = read_training_images<Container, Image>(folder, training_limit, [] { return Image(1, 28, 28); });
    dataset.training_labels = read_training_labels<Container, Label>(folder, training_limit);

    dataset.test_images = read_test_images<Container, Image>(folder, test_limit, [] { return Image(1, 28, 28); });
    dataset.test_labels = read_test_labels<Container, Label>(folder, test_limit);

    return dataset;
}


template <template <typename...> class Container, typename Image, typename Label = uint8_t>
MNIST_dataset<Container, Image, Label> read_dataset_3d(std::size_t training_limit = 0, std::size_t test_limit = 0) {
    return read_dataset_3d<Container, Image, Label>("mnist", training_limit, test_limit);;
}


template <template <typename...> class Container, typename Image, typename Label = uint8_t>
MNIST_dataset<Container, Image, Label> read_dataset_direct(const std::string& folder, std::size_t training_limit = 0, std::size_t test_limit = 0) {
    MNIST_dataset<Container, Image, Label> dataset;

    dataset.training_images = read_training_images<Container, Image>(folder, training_limit, [] { return Image(1 * 28 * 28); });
    dataset.training_labels = read_training_labels<Container, Label>(folder, training_limit);

    dataset.test_images = read_test_images<Container, Image>(folder, test_limit, [] { return Image(1 * 28 * 28); });
    dataset.test_labels = read_test_labels<Container, Label>(folder, test_limit);

    return dataset;
}


template <template <typename...> class Container, typename Image, typename Label = uint8_t>
MNIST_dataset<Container, Image, Label> read_dataset_direct(std::size_t training_limit = 0, std::size_t test_limit = 0) {
    return read_dataset_direct<Container, Image, Label>("mnist", training_limit, test_limit);
}


template <template <typename...> class Container = std::vector, template <typename...> class Sub = std::vector, typename Pixel = uint8_t, typename Label = uint8_t>
MNIST_dataset<Container, Sub<Pixel>, Label> read_dataset(std::size_t training_limit = 0, std::size_t test_limit = 0) {
    return read_dataset_direct<Container, Sub<Pixel>>(training_limit, test_limit);
}


template <template <typename...> class Container = std::vector, template <typename...> class Sub = std::vector, typename Pixel = uint8_t, typename Label = uint8_t>
MNIST_dataset<Container, Sub<Pixel>, Label> read_dataset(const std::string& folder, std::size_t training_limit = 0, std::size_t test_limit = 0) {
    return read_dataset_direct<Container, Sub<Pixel>>(folder, training_limit, test_limit);
}

}


#endif // READERMNIST_H
