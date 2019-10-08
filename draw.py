def draw(p1, display=False): 
    import matplotlib.pyplot as plt
    plt.figure('Draw')
    plt.plot(p1)
    plt.savefig("easyplot01.png")
    if display:
        plt.draw()  
        plt.pause(5)  
    plt.close()
