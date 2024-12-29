import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from harit_model.config.core import config
from harit_model.processing.data_manager import load_dataset, save_pipeline
from harit_model.processing.validation import evaluate_model
from harit_model.pipeline import train_mobilenetv2
from harit_model.processing.features import train_test_valid

def run_training() -> None:
    """Train the model."""
    
    # Download dataset
    load_dataset()
    
    # Prepare data
    _, train_data, test_data, valid_data, num_classes = train_test_valid(
        config.app_config.data_dir,
        target_size=(224, 224),
        batch_size=config.model_config.batch_size
    )
    
    # Create and train model
    model = train_mobilenetv2(num_classes)
    print("Training the MobileNetV2 model...")
    model.fit(
        train_data,
        validation_data=valid_data,
        epochs=config.model_config.epochs,
        batch_size=config.model_config.batch_size
    )

    # Evaluate model
    test_loss, test_acc = evaluate_model(model, test_data)
    
    # Save trained model
    save_pipeline(model)
    
    # Print results
    print(f'Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}')

if __name__ == "__main__":
    run_training()