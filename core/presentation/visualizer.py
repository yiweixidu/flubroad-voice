import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from io import BytesIO
from typing import Dict, List

def create_neutralization_heatmap(data: Dict[str, List[float]], title: str = "Neutralization Breadth") -> BytesIO:
    """生成中和广度热图，返回BytesIO图像对象"""
    df = pd.DataFrame(data).set_index("antibody")
    plt.figure(figsize=(6, max(4, len(df)*0.5)))
    sns.heatmap(df, annot=True, cmap="viridis", fmt=".2f", cbar_kws={'label': 'IC50 (µg/mL)'})
    plt.title(title)
    plt.tight_layout()
    img_bytes = BytesIO()
    plt.savefig(img_bytes, format='png', dpi=150)
    plt.close()
    img_bytes.seek(0)
    return img_bytes