
##### Placeholder
```bash
git clone https://github.com/{username}/{repo}
git submodule update --init --recursive

```
Follow BaM instructions


```
# unbind
sudo sh -c 'echo "0000:06:00.0" > /sys/bus/pci/devices/0000:06:00.0/driver/unbind'
cd bam build module
make load



sudo make unload
echo -n "0000:06:00.0" | sudo tee /sys/bus/pci/drivers/nvme/bind

# preconditioning.  2x
sudo dd if=/dev/urandom of=/dev/nvme2n1 bs=1G oflag=direct status=progress

sudo sh -c 'echo "0000:06:00.0" > /sys/bus/pci/devices/0000:06:00.0/driver/unbind'
make load


```