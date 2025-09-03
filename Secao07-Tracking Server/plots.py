import matplotlib.pyplot as plt



def plot_errors(y_true, y_pred):
    y_pred = y_pred.reshape(-1, 1)
    y_true = y_true.reshape(-1, 1)
    residuals = y_true - y_pred
    print(residuals.shape, y_true.shape, y_pred.shape)
    # ---------- Gráfico 1: Previsão vs Real ----------
    plt.figure(figsize=(6, 5))
    plt.scatter(y_true, y_pred, color="blue", label="Predições")
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", label="Ideal")
    plt.xlabel("Valor Real")
    plt.ylabel("Valor Previsto")
    plt.title("Previsão vs Real")
    plt.legend()
    plt.savefig("plots/prediction_vs_real.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ---------- Gráfico 2: Resíduos ----------
    plt.figure(figsize=(6, 5))
    plt.scatter(y_pred, residuals, color="purple")
    plt.axhline(y=0, color="r", linestyle="--")
    plt.xlabel("Valor Previsto")
    plt.ylabel("Resíduo")
    plt.title("Resíduos (y_true - y_pred)")
    plt.savefig("plots/residuos.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ---------- Gráfico 3: Distribuição dos Resíduos ----------
    plt.figure(figsize=(6, 5))
    plt.hist(residuals, bins=5, color="orange", edgecolor="black")
    plt.xlabel("Resíduo")
    plt.ylabel("Frequência")
    plt.title("Distribuição dos Resíduos")
    plt.savefig("plots/hist_residuos.png", dpi=300, bbox_inches="tight")
    plt.close()


