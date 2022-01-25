import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
from dataset import ImmitationDataSet
from torch.utils.data import DataLoader
set = ImmitationDataSet("val.csv")
loader = DataLoader(set, 4, num_workers=0)

for i in range(2):
    for _ in loader:
        pass