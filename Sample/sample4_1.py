#input

"""
print('PleaseEnterKey------ls')
str = input()
print(str)
"""

"""
words = ['keito','maho','tiara']
for w in words:
    print(w, len(w))
"""

"""
for i in range(5):
    print(i)
"""

#function
"""
def ask_ok(prompt, retries=4, reminder='Please try again'):
    while True:
        ok = input(promt)
        if ok in('y', 'ye', 'yes'):
            return True
        if ok in('n', 'no', 'nop'):
            return False
        retries -= 1
        if retries < -1:
            raise ValueError('invaild user response')
        print(reminder)

ask_ok('Do you want to quit?')
"""

"""
#class sample
class Spam:
    def __init__(self,ham,egg):
        self.ham = ham
        self.egg = egg

    def output(self):
        sum = self.ham + self.egg
        print("{0}".format(sum))

    val = 100
    def ham(self):
        self.egg('call method')

    def egg(self, msg):
        print("{0}".format(msg))
        print("{0}".format(self.val))


spam = Spam(5, 10)
spam.output()
"""
