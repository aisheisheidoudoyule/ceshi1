from tools import CorpusPreprocess, VectorEvaluation

if __name__ == "__main__":

        vec_eval = VectorEvaluation("output/glove.txt")
        vec_eval.get_similar_words("good")