direction: right

Translation project: {
  model implementation: {
    training input: {
      표준어 <-> 사투리: aihub
      표준어 -> translation model
      translation model -> 영어
    }

    training input -> model: encoder & decoder
    model: {
      baseline: {
        vanilla transformer
      }
      bilingual model: {
        ko-en GPT2
        ko-en T5
        ko-en BART
      }
    }
  }

  web app: {
    front-end: {
      design
      structure
    }

    back-end: {
      uWSGI
      localhost
    }
  }

  model implementation -> web app
}

