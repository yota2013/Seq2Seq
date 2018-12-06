import pickle

picklefilename="Vocabulary.pkl"
c_i = pickle.load(open(picklefilename, "rb"))

print(c_i)
print(len(c_i))


picklefilename="Vocabularytrain.pkl"
c_i = pickle.load(open(picklefilename, "rb"))

print(c_i)
print(len(c_i))

