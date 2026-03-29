"""Full curriculum training runner."""
import sys
sys.path.insert(0, '.')
from training.train import Trainer

trainer = Trainer(model_dir="models", log_dir="logs")
final = trainer.train_curriculum()
print(f"\nDONE: {final}")
