"""
2025/11/14
author: @kiuu_4

本プログラムの目的は、デュエマというゲームにおける先手後手の要素と運と実亮の要素の大きさを足らかにすることである。
そのために、デュエマの勝敗のデータからロジスティック回帰モデルを構築する。

目的を達成するために本スクリプトは以下の機能を有する。

1. DTLの試合結果一覧をDataFrameとして読み込む機能。
2. 読み込んだDataFrameから先手勝率を計算する機能。
3. 試合結果から各選手のレーティングを計算する機能。
4. ロジスティック回帰モデルを構築する為のデータ変換機能
5. 計算した各項目を用いて、ロジスティック回帰モデルを構築する機能


- 動作環境について
pyhon3.11.x
pandas 2.3.3
statsmodels 0.14.5
"""


"""
変動係数
レートの更新式で使用
"""
COEFFICIENT_VARIATION = 60

"""
定数定義
"""
MATCH_WIN = 1
MATCH_LOSE = 0


from logging import getLogger, FileHandler, StreamHandler, DEBUG, Formatter

def setup_logger(log_file="analize_DTL.log"):
    """
    ログ設定
    
    """
    
    # ロガー作成
    logger = getLogger(__name__)
    logger.setLevel(DEBUG)
    
    # ハンドラ作成（コンソール出力）
    handler = StreamHandler()
    handler.setLevel(DEBUG)

    # ファイル出力
    file_handler = FileHandler(log_file, mode='w', encoding="utf-8")
    file_handler.setLevel(DEBUG)
    
    
    # フォーマット設定
    formatter = Formatter(
        "%(asctime)s [%(levelname)s]    %(filename)s    %(funcName)s    message    %(message)s",
        "%Y/%m/%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # ハンドラをロガーに追加
    logger.addHandler(handler)
    logger.addHandler(file_handler)

    logger.propagate = False


    return logger




import pandas as pd

def loadDtlResult(logger):
    """
    DTL試合結果一覧読み込み処理
    DTL_match_result.tsv ファイルを読み込んでDataFrameを作成する
    先手勝利フラグを生成する
    
    inout
        logger: ログオブジェクト

    return
        df_DtlResult: デュエマの試合結果を扱うデータフレーム
    
    """

    logger.debug("処理開始: DTL試合結果一覧読み込み処理")

    df_DtlResult = pd.read_csv("./DTL_match_result.tsv", sep="\t")

    logger.debug(f"TSVファイル読み込み完了: {len(df_DtlResult)}件")
    
    # 先手勝利フラグ生成
    df_DtlResult["FIRST_WIN_FLG"] = df_DtlResult.apply(lambda x: 1 if x["FIRST_PLAYER"] == x["WIN_PLAYER"] else 0, axis=1).astype(int)


    logger.debug("返却データフレーム出力")
    logger.debug("\n%s", df_DtlResult.to_string())

    
    logger.debug("正常終了: DTL試合結果一覧読み込み処理")
    
    return df_DtlResult



def calcFirstWinRate(logger, df_DtlResult):
    """
    先手勝率計算処理
    input
        logger: ログオブジェクト
        df_DtlResult: DTL試合結果データフレーム
    
    return
        first_win_rate: 先手勝率（少数第4位を四捨五入）
    """

    logger.debug("処理開始: 先手勝率計算処理")

    
    total_first_win = df_DtlResult["FIRST_WIN_FLG"].sum()
    total_matches = len(df_DtlResult)

    # 四捨五入
    _FIRST_WIN_RATE = round(total_first_win / total_matches, 3)

    logger.debug(f"計算結果出力: 先手勝率 = {_FIRST_WIN_RATE}")

    logger.debug("正常終了: 先手勝率計算処理")


    return _FIRST_WIN_RATE



def calcRatingForEachPlayer(logger, df_DtlResult, _FIRST_WIN_RATE):
    """
    各選手のレーティングを計算する

    input
        logger: ログオブジェクト
        df_DtlResult: DTL試合結果データフレーム
        _FIRST_WIN_RATE: 先手勝率
    return
        df_Rating: レートを計算したデータフレーム, PAYER_NAME, RATING の2つのカラムを持つ

    """

    logger.debug("処理開始: レーティング計算処理")


    # 初期化処理, 各選手のレートを格納した辞書を作る
    dict_ratingForEachPlayer = {}

    for _, match in df_DtlResult.iterrows():
        if match["FIRST_PLAYER"] not in dict_ratingForEachPlayer.keys():
            dict_ratingForEachPlayer[match["FIRST_PLAYER"]] = 1500

        if match["SECOND_PLAYER"] not in dict_ratingForEachPlayer.keys():
            dict_ratingForEachPlayer[match["SECOND_PLAYER"]] = 1500
    

    logger.debug("初期化処理完了: プレイヤーの一覧と初期レート出力")
    logger.debug("\n%s", str(dict_ratingForEachPlayer))


    # レート計算処理
    for _, match in df_DtlResult.iterrows():
        """
        先手の期待勝率を計算する
        """
        # 先手プレイヤー, 後手プレイヤー
        first_player, second_player = match["FIRST_PLAYER"], match["SECOND_PLAYER"]
        
        # 先手のレート, 後手のレート
        rating_first_player, rating_second_player = dict_ratingForEachPlayer[first_player], dict_ratingForEachPlayer[second_player]
        
        # 先手の勝率を計算
        win_rate_expectation = calcWinRateExpectByRating(rating_first_player, rating_second_player, _FIRST_WIN_RATE)


        """
        レートの更新処理
        """
        # 先手勝ちの場合
        if match["FIRST_WIN_FLG"] == 1:
            # 先手勝ちでのレート更新
            dict_ratingForEachPlayer[first_player] = calcRatingAfter(rating_first_player, MATCH_WIN, win_rate_expectation)
            # 後手負けでメート更新
            dict_ratingForEachPlayer[second_player] = calcRatingAfter(rating_second_player, MATCH_LOSE, 1 - win_rate_expectation)

        # 後手勝ちの場合
        else:
            # 先手負けでレート更新
            dict_ratingForEachPlayer[first_player] = calcRatingAfter(rating_first_player, MATCH_LOSE, win_rate_expectation)
            # 後手勝ちでメート更新
            dict_ratingForEachPlayer[second_player] = calcRatingAfter(rating_second_player, MATCH_WIN, 1 - win_rate_expectation)
        

        year = match["YEAR"]
        section = match["SECTION"]
        match_no = match["MATCH"]

        logger.debug(f"【{year}|{section}|{match_no}】レート更新 - {first_player}: {rating_first_player} -> {dict_ratingForEachPlayer[first_player]}")
        logger.debug(f"【{year}|{section}|{match_no}】レート更新 - {second_player}: {rating_second_player} -> {dict_ratingForEachPlayer[second_player]}")


    logger.debug("レート計算処理完了")

    
    # データフレーム変換
    df_Rating = pd.DataFrame(list(dict_ratingForEachPlayer.items()), columns=["PLAYER_NAME", "RATING"])
    
    # レートの降順でソート
    df_Rating = df_Rating.sort_values(by="RATING", ascending=False).reset_index(drop=True)
    
    logger.debug("返却データフレーム出力")
    logger.debug("\n%s", df_Rating.to_string())

    
    logger.debug("正常終了: レーティング計算処理")

    return df_Rating



def calcWinRateExpectByRating(rating_myself, rating_opponent, _FIRST_WIN_RATE):
    """
    レートと先手勝率を考慮した、期待勝率計算処理

    input
        rating_myself: 自分自身のレート
        rating_opponent: 対戦相手のレート
        _FIRST_WIN_RATE: 先手勝率
    
    return
        win_rate_expectation: 期待勝率（補正後）


    期待勝率の計算式
    今回のレート導出にあたって、先手後手効果を考慮する。つまり、先手の方が期待勝率が高くなるよう補正を行う。
    これにより、後手で勝った時のレートの上り幅 > 先手で勝った時のレートの上り幅、先手で負けた時のレートの下がり幅 > 後手で負けた時のレートの下がり幅 となる。
    
    W_eAdjust = W_e*P_first / {W_e*P_first + (1-W_e)(1-P_first)}
    W_e= 1 / [1 + 10^{(R_opponent - R_myself)/400}]

    W_eAdjust: 期待勝率（先手後手補正後）
    W_e: 期待勝率（補正前）
    P_first: 先手勝率
    R_opponent: 対戦相手のレート
    R_myself: 自分のレート
    
    """
    # 補正前の期待勝率を計算
    win_rate_expectation = 1 / (1 + 10 ** ((rating_opponent - rating_myself) / 400))
    
    # 先手後手を補正
    win_rate_expectation = (win_rate_expectation*_FIRST_WIN_RATE)/((win_rate_expectation*_FIRST_WIN_RATE) + (1 - win_rate_expectation)*(1 - _FIRST_WIN_RATE))
    
    
    return win_rate_expectation



def calcRatingAfter(rating_before, MATCH_RESULT, win_rate_expectation):
    """
    試合後のレーティングを計算する処理

    レート更新式
    R_after = R_before + K * (W - W_eAdjust)

    R_after: 試合後のレート
    R_before: 試合前のレート
    K: 変動係数
    W: 勝敗（勝ち: 1, 負け: 0）
    W_eAdjust: 期待勝率（先手後手補正後）
    
    """

    return round(rating_before + COEFFICIENT_VARIATION*(MATCH_RESULT - win_rate_expectation), 1)




from sklearn.preprocessing import StandardScaler

def convertDataForLogisticRegression(logger, df_DtlResult, df_Rating):
    """
    ロジスティック回帰モデルを構築する為のデータ変換処理

    input
        logger: ログオブジェクト
        df_DtlResult: DTL試合結果データフレーム
        df_Rating: レートを計算したデータフレーム, PAYER_NAME, RATING の2つのカラムを持つ
    return
        df_forLogisticRegression: ロジスティック回帰を行うためのデータフレーム
    
    """

    logger.debug("処理開始: ロジスティック回帰モデルを構築する為のデータ変換処理")
    

    # 返却用のデータフレームを定義
    df_forLogisticRegression = pd.DataFrame(columns=["YEAR","SECTION","MATCH","PLAYER_NAME","MATCH_RESULT","JANKEN_RESULT"])
    
    logger.debug("空のデータフレーム出力")
    logger.debug("\n%s", df_forLogisticRegression.to_string())


    # マッチの結果とじゃんけんの結果を作成
    for _, match in df_DtlResult.iterrows():

        year = match["YEAR"]
        section = match["SECTION"]
        match_no = match["MATCH"]


        # 先手の処理後、後手を処理
        for player in [match["FIRST_PLAYER"], match["SECOND_PLAYER"]]:
            
            match_result = 1 if player == match["WIN_PLAYER"] else 0
            janken_result = 1 if player == match["FIRST_PLAYER"] else 0
            
            df_forLogisticRegression.loc[len(df_forLogisticRegression)] = [year, section, match_no, player, match_result, janken_result]
            
            
            
    # レーティングは標準化してからマージ
    scaler = StandardScaler()
    df_Rating_scaled = df_Rating.copy()
    df_Rating_scaled["RATING"] = scaler.fit_transform(df_Rating_scaled[["RATING"]])
    
    df_forLogisticRegression = pd.merge(
        left = df_forLogisticRegression,
        right = df_Rating_scaled,
        on = ["PLAYER_NAME"],
        how = "left"
    )


    logger.debug("返却データフレーム出力")
    logger.debug("\n%s", df_forLogisticRegression.to_string())


    logger.debug("正常終了: ロジスティック回帰モデルを構築する為のデータ変換処理")

    return df_forLogisticRegression




import statsmodels.api as sm

def buildLogisticRegressionModel(logger, df_forLogisticRegression):
    """
    ロジスティック回帰モデル構築処理

    input
        logger: ログオブジェクト
        df_forLogisticRegression: ロジスティック回帰を行うためのデータフレーム

    ロジスティック回帰モデル（勝敗がベルヌーイ分布に従う仮定）
    
    D = 1 / {1 + e^(-y)}
    y = a*J + b*R + l
    
    D: デュエマの勝敗（勝ち: 1, 負け: 0）
    J: じゃんけんの勝敗（勝ち: 1, 負け: 0）
    R: プレイヤーの技量
    l: 誤差（ここでは先手後手効果と技量だけで説明できない効果という意味で「運」と定義する）
    a,b: 定数
    
    """

    logger.debug("処理開始: ロジスティック回帰モデル構築処理")
    

    # 説明変数と目的変数
    X = df_forLogisticRegression[["JANKEN_RESULT", "RATING"]]
    y = df_forLogisticRegression["MATCH_RESULT"]

    # 定数項を追加
    X = sm.add_constant(X)

    
    # モデル構築、訓練
    model = sm.Logit(y, X)
    result = model.fit()

    
    logger.debug("モデル構築完了: 結果を表示")
    logger.info(result.summary())


    # 寄与率を計算して出力
    params = result.params
    abs_coef = params.abs()
    contribution_rate = abs_coef / abs_coef.sum()

    logger.debug("寄与率を計算, 結果を表示")
    for feature, rate in contribution_rate.items():
        logger.debug(f"{feature}: {rate:.2%}")

    
    logger.debug("正常終了: ロジスティック回帰モデル構築処理")

    return






def analize_DTL():
    """
    メイン処理
    """
    
    logger = setup_logger()
    logger.debug("処理開始: メイン処理")

    logger.debug(f"変動係数読み込み: COEFFICIENT_VARIATION = {COEFFICIENT_VARIATION}")


    """
    データフレーム読み込み
    """
    df_DtlResult = loadDtlResult(logger)


    """
    先手勝率計算処理
    """
    _FIRST_WIN_RATE = calcFirstWinRate(logger, df_DtlResult)


    """
    レート計算処理
    """
    df_Rating = calcRatingForEachPlayer(logger, df_DtlResult, _FIRST_WIN_RATE)


    """
    ロジスティック回帰モデルを構築する為のデータ変換処理
    """
    df_forLogisticRegression = convertDataForLogisticRegression(logger, df_DtlResult, df_Rating)


    """
    ロジスティック回帰モデル構築処理
    """
    buildLogisticRegressionModel(logger, df_forLogisticRegression)



    logger.debug("正常終了: メイン処理")
    
    return






if __name__ == "__main__":

    analize_DTL()



