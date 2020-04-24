exclude_songlist = [
    'AmarLal_Rest',
    'AmarLal_SpringDay1',
    'BrandonWebster_DontHearAThing',
    'BrandonWebster_YesSirICanFly',
    'ClaraBerryAndWooldog_TheBadGuys',
    'Debussy_LenfantProdigue',                   # classics
    'EthanHein_1930sSynthAndUprightBass',        # nothing to learn
    'EthanHein_BluesForNofi',                    # nothing to learn
    'EthanHein_GirlOnABridge',                   # nothing to learn
    'EthanHein_HarmonicaFigure',                 # nothing to learn
    'Handel_TornamiAVagheggiar',
    'JoelHelander_Definition',
    'JoelHelander_ExcessiveResistancetoChange',
    'JoelHelander_IntheAtticBedroom',
    'LizNelson_Coldwar',
    'LizNelson_ImComingHome',
    'LizNelson_Rainfall',
    'MatthewEntwistle_AnEveningWithOliver',
    'MatthewEntwistle_FairerHopes',
    'MatthewEntwistle_ImpressionsOfSaturn',
    'MatthewEntwistle_Lontano',
    'MatthewEntwistle_TheArch',
    'MatthewEntwistle_TheFlaxenField',
    'MichaelKropf_AllGoodThings',
    'Mozart_BesterJungling',
    'Mozart_DiesBildnis',
    'MusicDelta_Beethoven',
    'MusicDelta_ChineseChaoZhou',                 # too few stems
    'MusicDelta_ChineseDrama',                    # too few stems
    'MusicDelta_ChineseHenan',                    # too few stems
    'MusicDelta_ChineseJiangNan',                 # too few stems
    'MusicDelta_ChineseXinJing',                  # too few stems
    'MusicDelta_ChineseYaoZu',                    # too few stems
    'MusicDelta_GriegTrolltog',
    'MusicDelta_InTheHalloftheMountainKing',
    'MusicDelta_Pachelbel',
    'MusicDelta_Vivaldi',
    'Phoenix_BrokenPledgeChicagoReel',            # too few stems
    'Phoenix_ColliersDaughter',                   # too few stems
    'Phoenix_ElzicsFarewell',                     # too few stems
    'Phoenix_LarkOnTheStrandDrummondCastle',      # too few stems
    'Phoenix_ScotchMorris',                       # too few stems
    'Phoenix_SeanCaughlinsTheScartaglen',         # too few stems
    'Schubert_Erstarrung',
    'Schumann_Mignon',
    'TablaBreakbeatScience_Animoog',              # nothing to learn
    'TablaBreakbeatScience_CaptainSky',           # too few stems
    'TablaBreakbeatScience_MiloVsMongo',          # too few stems
    'TablaBreakbeatScience_MoodyPlucks',          # too few stems
    'TablaBreakbeatScience_PhaseTransition',      # too few stems
    'TablaBreakbeatScience_RockSteady',           # too few stems
    'TablaBreakbeatScience_Scorpio',              # too few stems
    'TablaBreakbeatScience_Vger',                 # too few stems
    'TablaBreakbeatScience_WhoIsIt',              # nothing to learn
    'Wolf_DieBekherte'
]

train_songlist = [
    'AClassicEducation_NightOwl',
    'AimeeNorwich_Child',
    'AimeeNorwich_Flying',
    'AlexanderRoss_GoodbyeBolero',
    'AlexanderRoss_VelvetCurtain',
    'Auctioneer_OurFutureFaces',
    'AvaLuna_Waterduct',
    'CelestialShore_DieForUs',
    'ClaraBerryAndWooldog_AirTraffic',
    'ClaraBerryAndWooldog_Boys',
    'ClaraBerryAndWooldog_Stella',
    'ClaraBerryAndWooldog_WaltzForMyVictims',
    'Creepoid_OldTree',
    'CroqueMadame_Oil',                 # instrumental
    'CroqueMadame_Pilot',               # instrumental
    'DreamersOfTheGhetto_HeavyLove',
    'FacesOnFilm_WaitingForGa',
    'FamilyBand_Again',
    'Grants_PunchDrunk',                # rap
    'HeladoNegro_MitadDelMundo',
    'HezekiahJones_BorrowedHeart',
    'HopAlong_SisterCities',
    'InvisibleFamiliars_DisturbingWildlife',
    'KarimDouaidy_Hopscotch',           # instrumental
    'KarimDouaidy_Yatora',              # instrumental
    'Lushlife_ToynbeeSuite',
    'MatthewEntwistle_DontYouEver',
    'NightPanther_Fire',
    'PortStWillow_StayEven',
    'PurlingHiss_Lolita',
    'SecretMountains_HighHorse',
    'Snowmine_Curfews',
    'StevenClark_Bounty',
    'StrandOfOaks_Spacestation',
    'SweetLights_YouLetMeDown',
    'TheDistricts_Vermont'
]

test_songlist = [
    'BigTroubles_Phantom',              # !
    'ChrisJacoby_BoothShotLincoln',     # instrumental
    'ChrisJacoby_PigsFoot',             # instrumental
    'Meaxic_TakeAStep',
    'Meaxic_YouListen',
    'MusicDelta_80sRock',
    'MusicDelta_Beatles',
    'MusicDelta_BebopJazz',
    'MusicDelta_Britpop',
    'MusicDelta_CoolJazz',              # instrumental
    'MusicDelta_Country1',
    'MusicDelta_Country2',
    'MusicDelta_Disco',
    'MusicDelta_FreeJazz',              # instrumental
    'MusicDelta_FunkJazz',              # instrumental
    'MusicDelta_FusionJazz',            # instrumental
    'MusicDelta_Gospel',
    'MusicDelta_Grunge',
    'MusicDelta_Hendrix',
    'MusicDelta_LatinJazz',             # instrumental
    'MusicDelta_ModalJazz',             # instrumental
    'MusicDelta_Punk',
    'MusicDelta_Reggae',
    'MusicDelta_Rock',
    'MusicDelta_Rockabilly',
    'MusicDelta_Shadows',               # instrumental
    'MusicDelta_SpeedMetal',            # instrumental
    'MusicDelta_SwingJazz',             # instrumental
    'MusicDelta_Zeppelin',              # instrumental
    'TheScarletBrand_LesFleursDuMal',   # terrible GT mix
    'TheSoSoGlos_Emergency'             # !
]

not_in_musdb18 = [
    'AimeeNorwich_Flying',
    'ChrisJacoby_BoothShotLincoln',
    'ChrisJacoby_PigsFoot',
    'ClaraBerryAndWooldog_Boys',
    'CroqueMadame_Oil',
    'CroqueMadame_Pilot',
    'FamilyBand_Again',
    'KarimDouaidy_Hopscotch',
    'KarimDouaidy_Yatora',
    'MusicDelta_BebopJazz',
    'MusicDelta_CoolJazz',
    'MusicDelta_FreeJazz',
    'MusicDelta_FunkJazz',
    'MusicDelta_FusionJazz',
    'MusicDelta_LatinJazz',
    'MusicDelta_ModalJazz',
    'MusicDelta_Shadows',
    'MusicDelta_SpeedMetal',
    'MusicDelta_SwingJazz',
    'MusicDelta_Zeppelin',
    'PurlingHiss_Lolita'
]