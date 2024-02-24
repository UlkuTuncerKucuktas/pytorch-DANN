import csv
import torch
import train
import mnist
import mnistm
import model

# Function to save results to a CSV file
def save_results_to_csv(results, filename):
    # Define the CSV file's column headers based on the expected result structure
    fieldnames = ['Augmentation', 'Training Mode', 'Source Correct', 'Source Total', 'Source Accuracy', 
                  'Target Correct', 'Target Total', 'Target Accuracy', 'Kannada Correct', 'Kannada Total', 
                  'Kannada Accuracy', 'Domain Correct', 'Domain Total', 'Domain Accuracy']

    # Write the results to a CSV file
    with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            # Flatten the result and adjust according to your actual results' structure
            flattened_result = {
                'Augmentation': result['augmentation'],
                'Training Mode': result['training_mode'],
                # Assuming 'result' is a dictionary with the required information
                'Source Correct': result['Source']['correct'],
                'Source Total': result['Source']['total'],
                'Source Accuracy': result['Source']['accuracy'],
                'Target Correct': result['Target']['correct'],
                'Target Total': result['Target']['total'],
                'Target Accuracy': result['Target']['accuracy'],
                'Kannada Correct': result['Kannada Unknown']['correct'],
                'Kannada Total': result['Kannada Unknown']['total'],
                'Kannada Accuracy': result['Kannada Unknown']['accuracy'],
                'Domain Correct': result.get('Domain', {}).get('correct', ''),
                'Domain Total': result.get('Domain', {}).get('total', ''),
                'Domain Accuracy': result.get('Domain', {}).get('accuracy', '')
            }
            writer.writerow(flattened_result)

def main():
    results = []  # This list will store all the results to be saved
    aug_dataset_dict = mnist.dataloader_dict
    target_train_loader = mnistm.mnistm_train_loader

    if torch.cuda.is_available():
        encoder = model.Extractor().cuda()
        classifier = model.Classifier().cuda()
        discriminator = model.Discriminator().cuda()
        for key, value in aug_dataset_dict.items():
            print('Training for augmentation ', key)
            result_list_source_only = train.source_only(encoder, classifier, value, target_train_loader)
            for result in result_list_source_only:
                result['augmentation'] = key
                result['training_mode'] = 'Source_only'
                results.append(result)
            result_list_dann = train.dann(encoder, classifier, discriminator, value, target_train_loader)
            for result in result_list_dann:
                result['augmentation'] = key
                result['training_mode'] = 'DANN'
                results.append(result)
    else:
        print("No GPUs available.")
    
    # Save the compiled results to a CSV file
    save_results_to_csv(results, 'training_results.csv')

if __name__ == "__main__":
    main()
