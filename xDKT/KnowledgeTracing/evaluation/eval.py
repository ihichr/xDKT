#eval.py  you can execute all files in google collab , each file is one cell
import sys
# sys.path.append('../')
import tqdm
import torch
import torch.nn as nn
from sklearn import metrics
from torch.autograd import Variable
#from Constant import Constants as C

def performance(ground_truth, prediction):
    print(f"[DEBUG] ground_truth size: {ground_truth.size()}, prediction size: {prediction.size()}")
    print(f"[DEBUG] unique values in ground_truth: {torch.unique(ground_truth)}")

    if ground_truth.numel() == 0:
        print("[ERREUR] ground_truth est vide. Évaluation sautée pour cette époque.")
        return

    if len(torch.unique(ground_truth)) < 2:
        print("[ERREUR] ground_truth contient une seule classe. Impossible de calculer ROC AUC.")
        return

    fpr, tpr, thresholds = metrics.roc_curve(
        ground_truth.cpu().detach().numpy(),
        prediction.cpu().detach().numpy()
    )
    auc = metrics.auc(fpr, tpr)

    f1 = metrics.f1_score(ground_truth.cpu().detach().numpy(), torch.round(prediction).cpu().detach().numpy())
    recall = metrics.recall_score(ground_truth.cpu().detach().numpy(), torch.round(prediction).cpu().detach().numpy())
    precision = metrics.precision_score(ground_truth.cpu().detach().numpy(), torch.round(prediction).cpu().detach().numpy())

    print('auc:' + str(auc) + ' f1: ' + str(f1) + ' recall: ' + str(recall) + ' precision: ' + str(precision) + '\n')

class lossFunc(nn.Module):
    def __init__(self):
        super(lossFunc, self).__init__()

    def forward(self, pred, batch):
      loss = torch.tensor(0.0, device="cuda", requires_grad=True)
      for student in range(pred.shape[0]):
          delta = batch[student][:,0:NUM_OF_QUESTIONS] + batch[student][:,NUM_OF_QUESTIONS:]
          temp = pred[student][:MAX_STEP - 1].mm(delta[1:].t())
          index = torch.LongTensor([[i for i in range(MAX_STEP - 1)]]).to("cuda")
          p = temp.gather(0, index)[0]
          a = (((batch[student][:, 0:NUM_OF_QUESTIONS] - batch[student][:, NUM_OF_QUESTIONS:]).sum(1) + 1)//2)[1:]
          for i in range(len(p)):
              if p[i] > 0:
                  loss = loss - (a[i]*torch.log(p[i]) + (1-a[i])*torch.log(1-p[i]))
      return loss

def train_epoch(model, trainLoader, optimizer, loss_func):
    for batch in tqdm.tqdm(trainLoader, desc='Training:    ', mininterval=2):
        #print(f"Debug: Training batch shape: {batch.shape}") #debug ==> torch.Size([64, 50, 100])
        batch = batch.to("cuda")
        pred = model(batch)
        loss = loss_func(pred, batch)
        optimizer.zero_grad()
        # if not loss.requires_grad or loss.grad_fn is None:
        #   print("[ERREUR] La perte n’est pas différentiable. Vérifiez loss_func.")
        #   print(f"[DEBUG] loss.requires_grad = {loss.requires_grad}")
        #   print(f"[DEBUG] loss.grad_fn = {loss.grad_fn}")
        # return model, optimizer  # ou raise une exception

        loss.backward()
        optimizer.step()
    return model, optimizer


def test_epoch(model, testLoader):
    device = next(model.parameters()).device
    gold_epoch = torch.empty(0, device=device)
    pred_epoch = torch.empty(0, device=device)
    print(f"[DEBUG] Taille testLoader: {len(testLoader)}")

    for batch in tqdm.tqdm(testLoader, desc='Testing:    ', mininterval=2):
        if len(batch.shape) == 2:
            batch = batch.unsqueeze(0)
        batch = batch.to(device)
        pred = model(batch)

        for student in range(pred.shape[0]):
            temp_pred = torch.empty(0, device=device)
            temp_gold = torch.empty(0, device=device)
            #print(f"[DEBUG] Étudiant {student} : delta contient des valeurs > 1")
            delta = batch[student][:, :NUM_OF_QUESTIONS] + batch[student][:, NUM_OF_QUESTIONS:]
            temp = pred[student][:MAX_STEP - 1].mm(delta[1:].t())
            p = temp.diag()

            a = (((batch[student][:, 0:NUM_OF_QUESTIONS] - batch[student][:, NUM_OF_QUESTIONS:]).sum(1) + 1) // 2)[1:]
            a = a.float()  # Important
# Debug sur les valeurs de p et a
            #print(f"[DEBUG] Étudiant {student}, prédictions p: {p}")
            #print(f"[DEBUG] Étudiant {student}, ground truth a: {a}")
            for i in range(len(p)):
                    # On garde seulement les valeurs de p entre 0 et 1 (évite les 0 et les 1 purs)
                    if 0 < p[i] < 1:
                        temp_pred = torch.cat([temp_pred, p[i:i+1].to(temp_pred.device)])
                        temp_gold = torch.cat([temp_gold, a[i:i+1].to(temp_gold.device)])

            pred_epoch = torch.cat([pred_epoch, temp_pred])
            gold_epoch = torch.cat([gold_epoch, temp_gold])

    return pred_epoch, gold_epoch

def train(trainLoaders, model, optimizer, lossFunc):
    for i in range(len(trainLoaders)):
        model, optimizer = train_epoch(model, trainLoaders[i], optimizer, lossFunc)
    return model, optimizer

def test(testLoaders, model):
    ground_truth = torch.Tensor([]).to("cuda")
    prediction = torch.Tensor([]).to("cuda")
    for loader in testLoaders:  # Utilisation directe de loader sans indexation
        pred_epoch, gold_epoch = test_epoch(model, loader)
        prediction = torch.cat([prediction, pred_epoch.to(prediction.device)])
        ground_truth = torch.cat([ground_truth, gold_epoch.to(ground_truth.device)])
    performance(ground_truth, prediction)
