import torch
import train
import mnist
import mnistm
import model


def main():
    aug_dataset_dict = mnist.dataloader_dict
    target_train_loader = mnistm.mnistm_train_loader

    if torch.cuda.is_available():
        encoder = model.Extractor().cuda()
        classifier = model.Classifier().cuda()
        discriminator = model.Discriminator().cuda()
        for key,value in aug_dataset_dict.items():
            print('Training for augmentation ' , key)
            result_list = train.source_only(encoder, classifier, value, target_train_loader)
            print(result_list)
            result_list =  train.dann(encoder, classifier, discriminator, value, target_train_loader)
            print(result_list)
        #train.source_only(encoder, classifier, source_train_loader_random_erase, target_train_loader)
        #train.dann(encoder, classifier, discriminator, source_train_loader, target_train_loader)
    else:
        print("No GPUs available.")


if __name__ == "__main__":
    main()
