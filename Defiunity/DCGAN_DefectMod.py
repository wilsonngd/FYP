import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from CocoDataset import CocoDataset

os.makedirs("images", exist_ok=True)

class DCGANTrainer:
    def __init__(self, data_path, ann_file_path, img_size=64, channels=1, latent_dim=100, n_epochs=10, batch_size=64, lr=0.0002, b1=0.5, b2=0.999, sample_interval=400, oversample='N', patience=10):
        self.data_path = data_path
        self.ann_path = ann_file_path
        self.img_size = img_size
        self.channels = channels
        self.latent_dim = latent_dim
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.sample_interval = sample_interval
        self.oversample = oversample
        self.patience = patience

        self.cuda = True if torch.cuda.is_available() else False

        # Initialize generator and discriminator
        self.generator = self.Generator(self)
        self.discriminator = self.Discriminator(self)
        self.adversarial_loss = torch.nn.BCELoss()

        # Load training dataset (replace with your specific dataset)
        self.dataset = self._load_dataset(data_path, self.ann_path)  # Function to be implemented

        # Create DataLoader for efficient training data access
        self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        if self.cuda:
            self.generator.cuda()
            self.discriminator.cuda()
            self.adversarial_loss.cuda()

        # Initialize weights
        self.generator.apply(self.weights_init_normal)
        self.discriminator.apply(self.weights_init_normal)
        
        # Optimizers
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        self.Tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor

        # Lists to track loss values
        self.g_losses = []
        self.d_losses = []
        self.val_f1_scores = []

    def _load_dataset(self, data_path, ann_file_path, caller='train'):
        # Define transformations
        transform=transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2,),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        # Create datasets
        if caller != 'val':
            dataset = CocoDataset(root_dir=data_path, ann_file=ann_file_path, transform=transform, oversample=self.oversample)
        else: # To avoid the validation data being oversampled
            dataset = CocoDataset(root_dir=data_path, ann_file=ann_file_path, transform=transform, oversample='N')

        return dataset

    def weights_init_normal(self, m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm2d") != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    class Generator(nn.Module):
        def __init__(self, outer_instance):
            super().__init__()

            self.init_size = outer_instance.img_size // 16  # 256 / 16 = 16
            self.l1 = nn.Sequential(nn.Linear(outer_instance.latent_dim, 128 * self.init_size ** 2))

            self.conv_blocks = nn.Sequential(
                nn.BatchNorm2d(128),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 128, 3, stride=1, padding=1),

                nn.BatchNorm2d(128, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 64, 3, stride=1, padding=1),
                
                nn.BatchNorm2d(64, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(64, 32, 3, stride=1, padding=1),
                
                nn.BatchNorm2d(32, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(32, outer_instance.channels, 3, stride=1, padding=1),

                nn.Tanh(),
            )

        def forward(self, z):
            out = self.l1(z)
            out = out.view(out.shape[0], 128, self.init_size, self.init_size)
            img = self.conv_blocks(out)
            return img

    class Discriminator(nn.Module):
        def __init__(self, outer_instance):
            super().__init__()

            def discriminator_block(in_filters, out_filters, bn=True):
                block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
                if bn:
                    block.append(nn.BatchNorm2d(out_filters, 0.8))
                return block

            self.model = nn.Sequential(
                *discriminator_block(outer_instance.channels, 16, bn=False),
                *discriminator_block(16, 32),
                *discriminator_block(32, 64),
                *discriminator_block(64, 128),
            )

            ds_size = outer_instance.img_size // 2 ** 4  # Downsampled size after conv layers
            self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())
            self.class_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

        def forward(self, img):
            out = self.model(img)
            out = out.view(out.shape[0], -1)
            validity = self.adv_layer(out)  # Real vs. Fake classification
            defectiveness = self.class_layer(out)  # Defect vs. Non-defect classification
            return validity, defectiveness

    def train(self, val_data_path, val_ann_file_path):
        best_val_metric = -np.inf  # Best validation metric (higher is better)
        early_stop_counter = 0  # Count epochs without improvement
        # Training loop
        for epoch in range(self.n_epochs):
            epoch_g_loss = 0
            epoch_d_loss = 0

            for i, (imgs, labels) in enumerate(self.data_loader):

                # Ground truths
                valid = Variable(self.Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
                fake = Variable(self.Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

                # Convert labels to Tensor 
                defect_labels = Variable(labels.type(self.Tensor).view(-1, 1), requires_grad=False)

                # Configure input
                real_imgs = Variable(imgs.type(self.Tensor))

                # -----------------
                #  Train Generator
                # -----------------

                self.optimizer_G.zero_grad()

                # Sample noise as generator input
                z = Variable(self.Tensor(np.random.normal(0, 1, (imgs.shape[0], self.latent_dim))))

                # Generate a batch of images
                gen_imgs = self.generator(z)

                # Loss measures generator's ability to fool the discriminator
                validity, _ = self.discriminator(gen_imgs)
                g_loss = self.adversarial_loss(validity, valid)

                g_loss.backward()
                self.optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                self.optimizer_D.zero_grad()

                # Measure discriminator's ability to classify real from generated samples
                real_validity, real_defectiveness = self.discriminator(real_imgs)
                fake_validity, _ = self.discriminator(gen_imgs.detach())

                real_loss = self.adversarial_loss(real_validity, valid)
                fake_loss = self.adversarial_loss(fake_validity, fake)

                # Classification loss for defectiveness
                defectiveness_loss = self.adversarial_loss(real_defectiveness, defect_labels)

                # Combine the losses
                alpha = 0.5  # Weight for adversarial loss
                beta = 0.5   # Weight for classification loss

                print("Fake loss: ", (real_loss + fake_loss) / 2, "Defect loss: ", defectiveness_loss)

                # Combine losses with weights
                d_loss = alpha * ((real_loss + fake_loss) / 2) + beta * defectiveness_loss

                d_loss.backward()
                self.optimizer_D.step()

                # Store loss values to track
                epoch_g_loss += g_loss.item()
                epoch_d_loss += d_loss.item()

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch + 1, self.n_epochs, i + 1, len(self.data_loader), d_loss.item(), g_loss.item())
                )

                batches_done = epoch * len(self.data_loader) + i
                if batches_done % self.sample_interval == 0:
                    save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)

            # Average losses over the epoch
            avg_g_loss = epoch_g_loss / len(self.data_loader)
            avg_d_loss = epoch_d_loss / len(self.data_loader)

            # Append to tracking lists
            self.g_losses.append(avg_g_loss)
            self.d_losses.append(avg_d_loss)

            print(
                f"[Epoch {epoch + 1}/{self.n_epochs}] [D loss: {avg_d_loss:.4f}] [G loss: {avg_g_loss:.4f}]"
            )

            # After each epoch, evaluate validation performance
            _, _, _, f1 = self.evaluate_discriminator(val_data_path, val_ann_file_path)
            val_metric = f1  # Choose F1 score or another metric as the validation metric
            self.val_f1_scores.append(val_metric)

            # Early stopping logic
            if val_metric > best_val_metric:
                best_val_metric = val_metric
                early_stop_counter = 0  # Reset early stop counter
                print(f"[Epoch {epoch + 1}] Validation Metric Improved: F1 Score = {val_metric:.4f}")
                self.save_model(self.discriminator, epoch=epoch + 1, best=True)
            else:
                early_stop_counter += 1
                print(f"[Epoch {epoch + 1}] No improvement for {early_stop_counter} epoch(s).")

            # Stop training if early stopping criteria are met
            if early_stop_counter >= self.patience:
                print("Early stopping triggered. Stopping training.")
                self.save_model(self.discriminator, epoch=epoch + 1, best=False)  # Save final model
                break

        print(f"Training completed. Best validation F1 Score: {best_val_metric:.4f}")

    def plot_losses(self):
        os.makedirs("Training result", exist_ok=True)
        # Plot the losses after training
        plt.figure(figsize=(10, 5))
        plt.plot(self.g_losses, label='Generator Loss')
        plt.plot(self.d_losses, label='Discriminator Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Losses Over Epochs')
        #plt.show()
        plt.savefig(os.path.join("Training result", "Loss.png"))
        plt.close()


    def save_model(self, discriminator, epoch, best=False, model_path="dcgan_models"):
        """
        Save the model with a filename indicating whether it's the best or final model.

        Args:
        - discriminator (torch.nn.Module): The discriminator model.
        - epoch (int): Current epoch, useful for versioning the model files.
        - best (bool): Whether the model is the best-performing model.
        - model_path (str): Directory to save the model files.
        """
        os.makedirs(model_path, exist_ok=True)
        if best:
            filename = f"{model_path}/best_discriminator.pth"
            print(f"Best model saved at epoch {epoch} to {filename}")
        else:
            filename = f"{model_path}/final_discriminator.pth"
            print(f"Final model saved at epoch {epoch} to {filename}")
        torch.save(discriminator.state_dict(), filename)

    def load_model(discriminator_class, model_path="dcgan_models"): 
        """
        Load the discriminator model from the specified path.

        Args:
        - discriminator_class (torch.nn.Module): The discriminator model class.
        - model_path (str): The path to the model file.

        Returns:
        - discriminator_model (torch.nn.Module): The loaded and evaluation mode model.
        """
        # Initialize the discriminator model
        discriminator_model = discriminator_class(None)  # Pass None for outer instance if not needed
        
        # Load the state_dict into the discriminator model
        try:
            # Check if CUDA is available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            discriminator_model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded discriminator model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
        
        # Set the model to evaluation mode for inference
        discriminator_model.eval()
        return discriminator_model
    
    def evaluate_discriminator(self, validation_data_path, ann_file_path, save_path="test_results"):

        val_dataset = self._load_dataset(validation_data_path, ann_file_path, caller='val')  # Function to be implemented

        # Create DataLoader for efficient training data access
        validation_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)

        # Ensure the save path exists
        os.makedirs(save_path, exist_ok=True)

        # Put the model in evaluation mode
        self.discriminator.eval()

        all_labels = []
        all_preds = []
        images_to_save = []

        with torch.no_grad():
            for imgs, labels in validation_dataloader:
                imgs = imgs.type(self.Tensor)  # Convert to the appropriate type

                # Predict on real images
                _, outputs = self.discriminator(imgs)
                preds = (outputs >= 0.5).cpu().numpy()

                all_labels.extend(labels.numpy())
                all_preds.extend(preds)

                # Save a subset of images with predictions
                if len(images_to_save) < 25:  # Save only the first 25 images for brevity
                    for img, pred, label in zip(imgs, preds, labels):
                        images_to_save.append((img, pred, label))
                        if len(images_to_save) >= 25:
                            break

        # Flatten lists for metrics calculation
        all_labels = np.array(all_labels).flatten()
        all_preds = np.array(all_preds).flatten()

        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)

        # Confusion Matrix
        cm = confusion_matrix(all_labels, all_preds)
        cm_display = ConfusionMatrixDisplay(confusion_matrix=cm)
        cm_display.plot(cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(save_path, "confusion_matrix.png"))
        plt.close()

        # Save images with predictions
        for i, (img, pred, label) in enumerate(images_to_save):
            img = (img * 0.5 + 0.5).cpu().numpy()  # Denormalize image
            img = np.transpose(img, (1, 2, 0))  # Convert from CHW to HWC

            plt.imshow(img)
            plt.title(f"Predicted: {'Defective' if pred else 'Non-Defective'} | Actual: {'Defective' if label else 'Non-Defective'}")
            plt.axis('off')
            plt.savefig(os.path.join(save_path, f"image_{i}.png"))
            plt.close()

        return accuracy, precision, recall, f1
