# APLICANDO REDES DE APRENDIZADO PROFUNDO E ALGORITMOS DE SELEÇÃO DINÂMICA PARA CLASSIFICAR IMAGENS DE CÂNCER DE PELE

Olá, boas-vindas ao repositório que contem os códigos utilizados no meu trabalho de conclusão de curso.
Este repositório contem os códigos de:

- Redes Neurais Convolucionais autuando como classificadores e extratoras
- Algoritmos classicos de aprendizado de máquina utilizados para classisficar os atributos extraidos pelas CNNs

Para este trabalho utilizamos o banco de imagens HAM10000, este banco pode ser encontrado neste [link](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000).

Para facilitar a utilização dos códigos, este repositório segue o esquema de pastas que utilizei no trabalho, fique a vontade para alterá-lo como preferir.

O passo a passo para a primeira utilização deste repositório esta descrito abaixo:

- Fazer o download do banco [HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- Colocar as imagens do banco na pasta /cancer/data
- Dentro da pasta /cancer é possivel encontrar o arquivo 'data.txt'. Este arquivo é estruturado da seguinte forma: Cada linha contem o nome de uma imagem (Ex: ISIC_0024306) e em sequência a classe que esta imagem pertence. Isso se repete para todas as imagens do banco.
- Para utilizar as CNNs como classificadoras, basta executar o arquivo 'CNNPredict.py'
- Antes de utilizar os métodos clássicos para classificação, devemos extrair os atributos, para isto devemos executar o arquivo 'FeatureExtraction.py'.
- É importante notar que as caracteristicas extraidas são encaminhadas para a pasta /cancer/libsvm e diferentes CNNs geram difentes nomes de arquivo, portanto antes de executar o código de aprendizado de máquina, devemos conferir se o caminho está correto.

  Uma prévia dos melhores resultados atingidos podem ser encontrados na tabela abaixo:
![Resultados](https://github.com/JoaoPedroCAS/ClassificarLes-oDePele/assets/70914320/0eb18656-4fb0-447c-bf64-20fe465b7593)


