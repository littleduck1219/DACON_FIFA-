# FIFA 선수 이적료 예측 경진대회
대회 자료 -> https://dacon.io/competitions/open/235538/overview/description

주어진 자료를 토대로 선수 이적료를 예측합니다.

작성 코드 : <a href="https://github.com/littleduck1219/DACON_FIFA_Player_Transfer_Fee_Predict/blob/main/FIFA_player_transfer_fee_predict.ipynb">FIFA_player_transfer_fee_predict.ipynb</a>

1. FIFA_train.csv / FIFA_test.csv\
 id : 선수 고유의 아이디\
 name : 이름\
 age : 나이\
 continent : 선수들의 국적이 포함되어 있는 대륙입니다\
 contract_until : 선수의 계약기간이 언제까지인지 나타내어 줍니다\
 position : 선수가 선호하는 포지션입니다. ex) 공격수, 수비수 등\
 prefer_foot : 선수가 선호하는 발입니다. ex) 오른발\
 reputation : 선수가 유명한 정도입니다. ex) 높은 수치일 수록 유명한 선수\
 stat_overall : 선수의 현재 능력치 입니다.\
 stat_potential : 선수가 경험 및 노력을 통해 발전할 수 있는 정도입니다.\
 stat_skill_moves : 선수의 개인기 능력치 입니다.\
 value : FIFA가 선정한 선수의 이적 시장 가격 (단위 : 유로) 입니다\


2. submission.csv (제출 파일 형식)\
id : 선수 고유의 아이디\
value : 예측된 선수 이적 시장 가격을 기입\ 
