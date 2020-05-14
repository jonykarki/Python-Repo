"Progress Bars using tqdm"
import tqdm
import random
import time

test_list = random.sample(range(100), 20)

tq = tqdm.tqdm(test_list, desc="Working")

# using enumerate
for i, item in enumerate(tq):
    time.sleep(0.1)
tq.close()

bar = tqdm.tqdm(total=1000, desc="Calculating")
# manually + using trange
for i in range(5):
    time.sleep(0.5)
    bar.update(200)
bar.close()