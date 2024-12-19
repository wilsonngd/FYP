import os
from datetime import datetime
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' #or can create a __init__.py to declare the package

# from DCGAN_ResNet import DCGANTrainer
from DCGAN_DefectMod import DCGANTrainer

if __name__ == "__main__":
    train_path = "C:/Users/User/OneDrive/Documents/Wilson/TARUMT/Degree/Year3/FYP_Dataset/DefectData/train"
    train_ann_file_path = "C:/Users/User/OneDrive/Documents/Wilson/TARUMT/Degree/Year3/FYP_Dataset/DefectData/train/_annotations.coco.json"
    validation_path = "C:/Users/User/OneDrive/Documents/Wilson/TARUMT/Degree/Year3/FYP_Dataset/DefectData/valid"
    val_ann_file_path = "C:/Users/User/OneDrive/Documents/Wilson/TARUMT/Degree/Year3/FYP_Dataset/DefectData/valid/_annotations.coco.json"
    test_path = "C:/Users/User/OneDrive/Documents/Wilson/TARUMT/Degree/Year3/FYP_Dataset/DefectData/test"
    test_ann_file_path = "C:/Users/User/OneDrive/Documents/Wilson/TARUMT/Degree/Year3/FYP_Dataset/DefectData/test/_annotations.coco.json"

    # Initialize DCGAN
    dcgan = DCGANTrainer(
        data_path=train_path,
        ann_file_path = train_ann_file_path,
        img_size=32,
        channels=3,
        n_epochs=50,
        batch_size=32,
        lr=0.0002,
        sample_interval=400,
        oversample='Y'
    )
    
    # Training
    start_time = datetime.now()
    print(f"Training started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    dcgan.train(validation_path, val_ann_file_path)
    dcgan.plot_losses()
    dcgan.plot_acc_scores()
    dcgan.plot_f1_scores()
    
    # Evaluate the discriminator
    accuracy, precision, recall, f1 = dcgan.evaluate_discriminator(test_path, test_ann_file_path)
    print(f"Discriminator Evaluation - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    end_time = datetime.now()
    print(f"Training ended at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    duration = end_time - start_time
    print(f"Training duration: {duration}")
