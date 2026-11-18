# Vision Transformer – reprodukcja wyniku

Projekt z przedmiotu Analiza danych obrazowych i multimedialnych.
Zespół: Oliwia Wołos, Julia Cwynar, Jan Krokos.

Reprodukcja artykułu A. Dosovitskiy i in., *An Image is Worth 16x16 Words*, ICLR 2021 ([arXiv:2010.11929](https://arxiv.org/abs/2010.11929)), kod autorów: [google-research/vision_transformer](https://github.com/google-research/vision_transformer).

Notatnik `Vision_Transformer.ipynb`:

- odtwarza krok po kroku działanie ViT: podział obrazu na patche i tokeny, self-attention, blok transformera i token CLS, i sprawdza, że własna implementacja daje ten sam wynik co `timm` (sekcje 5–7),
- doucza ViT-B/32 i ViT-B/16 na CIFAR-10 i CIFAR-100 protokołem autorów i porównuje wyniki z Tabelą 5 artykułu (sekcje 8–10).

## Wyniki (accuracy na zbiorze testowym)

| Model | Zbiór | Artykuł | Nasz wynik | Różnica |
|---|---|---|---|---|
| ViT-B/32 | CIFAR-10 | 98,79% | 98,56% | −0,23 pp |
| ViT-B/32 | CIFAR-100 | 91,97% | 91,37% | −0,60 pp |
| ViT-B/16 | CIFAR-10 | 98,95% | 98,85% | −0,10 pp |
| ViT-B/16 | CIFAR-100 | 91,67% | 92,78% | +1,11 pp |

Odstępstwa od artykułu: 500 zamiast 10 000 kroków i 224 zamiast 384 px (czas obliczeń). Wyjaśnienie różnic: sekcja 10.1 notatnika.

## Uruchomienie

Środowisko: Python 3.10, GPU NVIDIA (testowane na RTX 4070 Ti SUPER 16 GB, CUDA 11.8).

```bash
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Następnie otworzyć notatnik i wykonać *Run All*. Przy `RETRAIN = False` wyniki treningu są wczytywane z folderu `cache/` (ok. 2 min); bez tego folderu notatnik trenuje modele od nowa (ok. 50 min).

Dane (CIFAR-10/100, repozytoria `uoft-cs/cifar10` i `uoft-cs/cifar100` na Hugging Face, przypięte wersje) i wagi (`timm`: `vit_base_patch{16,32}_224.orig_in21k`, te same checkpointy co `google/vit-base-patch{16,32}-224-in21k`) pobierają się automatycznie i nie są przechowywane w repozytorium.
