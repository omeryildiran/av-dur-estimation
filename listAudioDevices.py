from psychopy import prefs
from psychopy.sound import backend_ptb as ptb
devices= ptb.getDevices(kind='output')
print('\nDevices\n')

for i in devices:
    print(devices[i]['DeviceIndex'], ' ', devices[i]['DeviceName'])
    #print(devices[i]['DeviceIndex'])
print('\n')
