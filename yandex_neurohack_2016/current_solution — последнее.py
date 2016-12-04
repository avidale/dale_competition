import numpy as np
import sys
from collections import deque

if sys.version_info.major == 2:
	input = raw_input

OFFSET = 3000
SKIP_SIZE = 28
TARGET_CHANNEL = 15
N_CHANNELS = 21

experiment_id = input()


common_coef = [0.31652657391336914, -0.11331214078666056, -0.22828638897616402, -0.11326814012435339, 0.38783418068825704, -0.06733356712839139, -0.03788484676716095, 0.2377981980975767, 0.08926536303058749, -0.07251694015192743, 0.23894185316412278, 0.42899163955163105, -0.2103824986272829, 0.345815744236666, 0.15383095975450986, -0.5525704044479368, 0.2664070609421818, 0.018169435867321785, -0.040183283154506734, 0.11917690585191054, 0.2237345416833917, -0.21047547904786595, 0.04594125246169171, 0.18922350370785984, -0.17696803791099006, -0.18668429040346374, -0.10230709279092423, -0.1812280055084521, -0.2692368685274582, -0.1573319178384639, -0.07120510594672122, -0.3255219091811454, -0.278863283373336, -0.25367088303997487, -0.3539194876317602, -0.5322250539983064, -0.2354413280134891, -0.3445594271466774, -0.30691892968196477, -0.13592379735829865, -0.13208864457671435, -0.3800049242628901, 0.002249676801370315, -0.014523434346058002, 0.12998682553656113, 0.39093262810139484, -0.18819734450042036, 0.1134714540907923, 0.4021807272574689, 0.1635533774748326, -0.05938763240791034, 0.20501673862328199, 0.17660857003127634, -0.15476818209577936, 0.09564473073980033, 0.01618886594779643, 0.24482125728693716, 0.4749411792689526, -0.13718034508937996, 0.27906783914891187, 0.0660418944396247, -0.11012306328581889, -0.039620634877020285, -0.012115414161187036, 0.16620192512378387, 0.016717796601513714, 0.21500290181797294, -0.07329053387190483, 0.10997083621748409, 0.1434353863700504, 0.007420090021996801, -0.33618875440253615, 0.2156228772454137, -0.20684540595117024, 0.005537070367490316, 0.3731620247735765, -0.11924813066585524, 0.2256310102650758, 0.7073896894663041, 0.08944638390694691, 0.2860126379159121, 0.053712313141439934, -0.1678055450278517, -0.05521665142639983, -0.17293025088644248, -0.22649421994839278, -0.34488364604910854, -0.5655917451056038, -0.20321307065890196, -0.19260115392478047, -0.3739566267793221, -0.2412283477310131, 0.4101603936687177, -0.25100729860324095, 0.29081446719722953, -0.059778134913039696, 0.4184401416055946, 0.24635447252358966, 0.08989813923001985, 0.5599728136665043, 0.07434581704841736, 0.03703083930922559, -0.10127069995080473, 0.20203557377973339, 0.4557633243599639, 0.14060131410893034, -0.17022439199319556, 0.21491208865736064, -0.01567844062837003, 0.28099089254371734, 0.12972310766125553, -0.16429670630695184, -0.026059061084991102, 0.45236716117136094, -0.10728377061389599, -0.007745231075707312, 0.21869876623569476, 0.17090761742709323, -0.1745748221624129, -0.14027049990466686, 0.304540202420457, 0.38088189146739443, -0.3316597989237834, 0.18317509952375358, 0.550886002483837, 0.39534059373657304, 0.30264411661180624, 0.10301493109634621, -0.0006537234752993559, 0.3337155730714272, 0.27725242692776403, 0.17397895760682922, 0.022855098141008254, 0.24991244176885993, -0.014973799981955742, -0.0173341811868483, -0.0498412876991484, 0.05376110869224065, -0.43756369970566894, 0.0657279019439978, -0.2575662490510943, -0.7661721537543621, -0.12280182765712738, 0.015284279033239933, 0.2498344459980108, -0.16712246675057726, -0.38951051744848014, -0.30342672311046076, 0.07853701635937341, 0.07573709177158114, -0.1636036634057219, -0.1412267702281484, -0.2860841962650094, 0.17295327276313832, 0.24696976805300008, -0.5864298341509877, -0.2809249597823621, -0.36016565418391916, -0.25626468863563323, -0.8741912681862506, -0.22699943085442806, 0.19251283261158078, -1.4952659245200386, -0.6602325661944901, -0.12819636560394967, -0.3777565657726285, -0.7349050787935234, -0.9105823020597379, 0.11449415579752655, 0.3425625695660441, 0.03998277153998725, 0.027023329566634988, 0.12490985710713452, 0.1551198877161992, -0.05743179822162252, -0.24287977975336736, 0.058976943715146585, 0.1806746390572476, 0.25305939001876415, 0.18177738431927326, 0.3505233275596339, 0.4103582379633981, -0.11851532228300161, 0.16730761621731202, 0.2790169425080299, 0.4991246813172579, -0.0063884043299171165, 0.265247932472283, 0.6615855017246196, 0.03772906739763157, -0.10091481575968785, -0.060855337365905475, -0.3380988594611492, -0.0873593759731744, -0.06480488022459745, -0.025327662374987493, 0.16319983863792978, -0.059829531470170554, 0.07412720800395148, 0.04902231415553435, 0.1231246249252694, 0.10557324878720516, -0.17559929578159142, 0.03457612309790351, 0.3560904035762406, 0.37811761346885536, -0.164057570525338, 0.00202894964430143, 0.17509201608011002, 0.35766974263652646, 0.03131040816864126, -0.09650137279042002, -0.5152647793949817, 0.3106341113038754, 0.0527157635536527, -0.15730105720593904, -0.0301187628976449, 0.07716496330407417, -0.030446693935483674, 0.14508289227958043, 0.02057991072582804, 0.1606576618150954, -0.07561303865105166, 0.03904904882800534, -0.033351293590009, 0.23296555703129077, 0.1422950783935658, -0.509744011522361, 0.11052939760915999, -0.03206919418084246, -0.20103311060210288, 0.085771713958338, -0.27258949851386166, 0.4097663961152375, 0.039185879447267966, 0.02925086462647432, -0.13046871585656755, -0.048930305579830836, -0.06527657355498742, 0.34033930481109326, 0.17485951339285305, 0.3097871280590786, -0.008964198510071734, -0.010560626861598255, -0.2858343782990165, 0.0342757265325166, -0.12150423423314406, -0.32746059434514496, -0.06960936397493134, -0.0819098966145214, -0.05438084768720028, -0.24186741526264682, 0.0597651403761904, -0.24181410150100538, 0.1279211573372768, 0.13439891205110086, 0.08468569650095638, 0.11376148023109851, 0.04475770440870065, -0.16490216336935046, -0.09033844373889102, -0.028452081608711998, 0.11821281779759475, -0.10656527726657349, 0.026635703181204833, 0.30538070517812843, 0.06302143885211745, 0.08517633904269813, -0.3252178574161503, 0.4525921070724107, 0.02539689049476694, 0.13134392953702145, 0.06229596191833658, -0.05758813708585593, -0.0299293014879507, -0.2553462478856281, -0.28309131240720753, -0.02616328837493488, 0.2259634689478625, 0.17879882540896141, 0.20910182058790872, -0.5416973539059367, -0.06974555348752265, -0.21596126313609842, 0.002502067986173884, 0.06161400197651262, 0.39253793015724714, -0.30196615583756925, 0.3304029312901847, -0.08437285580460525, 0.3581859176531614, -0.1674366719411902, 0.16319492314598028, 0.15261311121853283, -0.04488650794434765, 0.55152973257736, 0.28428826475721786, -0.05293224503969371, -0.04696189358803295, 0.07845373849152994, -0.17077341323272258, -0.01601844296121051, 0.26474724154948726, -0.18629623396181413, -0.11730994382309289, 0.03293765970788861, 0.07153143575880262, -0.23410692961333773, 0.19529251531559413, 0.4391272241694285, 0.20136924804343617, -0.280778438020141, -0.18930368647640183, -0.0461060412649606, 0.11067562445601777, 0.0340325847272017, -0.07994557241145356, -0.2947102353275718, -0.14732052141921453, -0.11227547389263312, -0.09654363998375953, -0.24173364345027232, -0.005620751249866845, 0.18790470444491478, -0.1337722417370288, -0.28738352726082667, 0.1013343279932223, -0.015029842601889143, -0.2627442234634078, 0.28976821124194224, 0.3136487877820574, 0.1997234757758783, -0.02288645032760607, 0.17666836713878517, -0.24330610362727337, -0.24659461439315483, -0.07368061991946633, -0.2658533641469554, 0.06652435438868529, 0.14828691707705, 0.006362030391702087, 0.04234159635613394, -0.07544908415528043, 0.06313004659206188, 0.010348639974704817, -0.15022915106112367, 0.0069971177627616, 0.11803394028602747, -0.07363793796726895, -0.058924687005858914, 0.0823546711285672, 0.11095449361714046, -0.024438566960153268, 0.13281349031050804, 0.1464315000660102, 0.05106200623042035, -0.11247507108774393, 0.021027048753597406, -0.07204055258793789, 0.1771295395571707, 0.08763482555306434, 0.04781888849492305, 0.09672772349500032, 0.204927539901911, 0.21698982098987338, -0.07942133270122971, -0.017923814408336396, 0.1919894021570165, -0.06793210342705552, 0.06334184734505123, 0.08913831477073605, -0.517023911779153, -0.14650560603850815, -0.017636766082830477, 0.18351190971939707, -0.19921333890925497, -0.12519964984751364, 0.13153712381490074, 0.047245714023529256, 0.27558774382690054, -0.2724489959622475, -0.051589247324760834, 0.10706762419413583, 0.037554385657960904, 0.1327091008524253, -0.10025302661390077, 0.20321318657753779, 0.19114986669891676, 0.29689900655260726, -0.16076727809274086, 0.17245765036438399, -0.049484402729433476, -0.19128111840670073, -0.17586991908410035, 0.041578367618913765, -0.03357047983732326, -0.07026831075603215, -0.09466718695732981, 0.19153462032711013, -0.13960938343895515, -0.03480341400611972, 0.140611832707807, -0.2064651414379901, 0.05136979651318433, -0.29582238240446423, -0.1587413293728664, -0.23553468457400412, -0.15864329024504992, 0.1535687049307653, -0.21474038407978785, 0.009594391758448586, -0.12141313483492346, 0.1106317071217611, 0.2835173362370259, -0.27967505593308756, -0.020330129427090074, -0.17654476693962015, 0.2686840854368274, 0.1981969169983918, -0.12299263684275837, 0.20507191528629246, -0.26206667946597023, 0.19998059279627436, 0.16374777688172504, 0.14265423988610357, -0.1334210737271072, 0.13100918657038166, 0.19150544415784718, -0.038067889145076424, -0.06562390628069148, 0.03899882095685725, 0.1482679877207065, -0.1869424400773689, 0.08085573846962586, 0.12729368254297174, -0.3637338976481629, -0.007619445245454206, 0.024799325244774195, 0.05964730536999639, 0.1579402078702515, -0.14238287051993329, 0.24020235831941655, -0.23100949663370507, -0.3099577454867725, 0.037271813771417577, 0.08083609734481732, 0.02953254957444532, -0.22608059827990307, 0.06017528087979059, 0.12065499513793634, -0.1880927843391945, 0.27043822533398554, -0.060062128867529554, -0.12525249244598835, -0.15239931500442133, -0.3026490103911265, -0.3465681495492526, -0.11853214762779359, 0.14155858684751393, 0.11154348230296143, -0.1030880978463044, 0.10422984169157785, 0.05753344891573073, 0.13035506381589115, -0.012122336875096078, 0.15328935972059246, 0.17486348776984104, 0.20712861899738094, 0.05141382049654187, 0.09923696248611757, -0.11874171163618262, 0.16054351918353885, -0.11663575793559826, 0.21256337351856025, 0.02225961392922689, 0.1575867817172313, 0.08944200017651158, 0.11107554405485065, 0.0077584034138753555, -0.15746845863055373, -0.09032398932643122, 0.06624719379807716, 0.3294343558929099, -0.24181302308050806, 0.06880931531276159, 0.07284903635522035, -0.3158801604344095, -0.1267362000507021, 0.02184171375419276, -0.08141929808846032, -0.11710112903278322, -0.16056534387353413, 0.008202679544509256, -0.2008872714850902, 0.14915183750695932, -0.1121490959595073, 0.2537649306225808, 0.07992440245428808, 0.21488096294236353, -0.14058140591637439, -0.06609415413552833, -0.39843361693721785, -0.049179764774387624, 0.003254604232508803, 0.009536061818544371, 0.1441153933133592, 0.08326750202082937, -0.0037485083988168352, -0.1321457999805102, 0.11204095746948428, 0.21909261886164488, 0.172382156816652, 0.02297354395069003, -0.0780616643246209, 0.04406802851346262, 0.11957639613590772, 0.08112911266261079, 0.003135485643157403, -0.191734685900449, 0.3831138046012479, -0.13356481861957303, 0.16150809227126167, -0.06444831847127115, 0.04365918112962215, -0.268987166240146, -0.02323457560623026, -0.2920167156973696, -0.30517928624446294, 0.004376035534635496, -0.07351911425210099, -0.04900641261331434, -0.21479471164609604, 0.026943083345624893, 0.00944126768267255, -0.07216101401206998, -0.1561654496125624, -0.10554533704923914, 0.022220389138208854, -0.4826587592103149, 0.13451803770133222, 0.3654101176741851, 0.0008180260417917226, 0.09425515532836477, 0.13050058436726117, -0.18738404934162528, -0.2962040591481818, 0.27921142266440063, -0.025247721857325864, 0.067271141529256, 0.015487120832185319, 0.21943835444790186, 0.059924411029798445, -0.30541185612720567, 0.12649823092422016, 0.27448313317243206, 0.30352589786313267, 0.032298529607275105, 0.07366903338850182, -0.0744184991855294, 0.22765780798854693, 0.1770606327203728, 0.44642625574175615, 0.32030099888002933, 0.27236070235962195, -0.03932524384888596, 0.01951939118047039, -0.005033930676375318, 0.03955272670131471, 0.05303617235645851, 0.0916811878099378, -0.10961006390593525, 0.05996229727383474, 0.018978381717593616, -0.1196555578374399, 0.13804854179390819, 0.01450094099133952, -0.10423540903896573, 0.17362531202044204, -0.02121840936582137, -0.18976748811312275, 0.29540758996640865, -0.023050991147373554, -0.04255774697318279, -0.09527196925379669, -0.11977935860331305, -0.20677618869088468, -0.01194089353835822, 0.1689117832360359, -0.18758983816765637, -0.2663243665846497, 0.15379925073645065, 0.019703288317717463, 0.011938314314998788, -0.08262341318850809, 0.03762816299224089, 0.14596388675814703, -0.275381863373963, -0.09668206384515554, -0.0005648510528769055, 0.1796598863119468, 0.05745070556347835, -0.12224801399272818, -0.3687716464200447, -0.06305880852335412, -0.06168546859931655, -0.05248578530734548, -0.07130142901473718, 0.36280918264025963, 0.10111034330033396, 0.08447053008991129, -0.13349638793838775, -0.23917219904576978, 0.24379510995969794, -0.030027561635886213, 0.04235201903221002, 0.027726000499203523, -0.05573367054680308, 0.18207253429672274, -0.11222419294444852, 0.09401330853522044, 0.1886096318145719, 0.34444636620776414, -0.06492012968280406, 0.02509772817354362, 0.2899846455833135, 0.023965355643065293, 0.1590115829214753, 0.05571639636184453, 0.022886525373332484, 0.044935556304404774, -0.1232482957032412, 0.08260938083864534, 0.03314224960378916, 0.04911293959338538, -0.09277168270074486, 0.1379037171886706, -0.039294776995790315, -0.1455232550476693, 0.032933806304183036, -0.29558889361924917, -0.0898923622232004, -0.37546327619517966, 0.15087984219391873, -0.160962628586504, 0.04077186169413282, 0.2967903168341801, 0.007405362902387428, -0.0517722397574836, 0.06521127528230627, -0.35366946386556186, 0.06439691062619683, -0.1400689015249017, 0.12771643852519987, 0.3345118074489239, -0.050964368475924, 0.19908565669784486, 0.3128643706087698, 0.15269998833358325, -0.17147707829502218, 0.08796167124327618, -0.04516226597400301, -0.04576288599703977, -0.10825362849481007, 0.03134629476946682, 0.14073713898655685, 0.2699438789800714, 0.13889780720794537, 0.18380264150051642, 0.22187316721475883, 0.11212039987626705, 0.03291656436406379, -0.039829301552055496, -0.28551558903239094, -0.23382155645536892, -0.1777187378097763, -0.3021427803995286, 0.049146905910532286, -0.12141420600531933, 0.16030115374485837, -0.2691660389398251, -0.040955102581651304, -0.06978333743292997, -0.1481243084941797, 0.13488701515039317, -0.3150925457351125, 0.01428969472660435, -0.19541310398014639, -0.02614378312966346, -0.10241906789059137, -0.008439787721538182, 0.03687360372151363, -0.05362844395234119, -0.062470471895354006, 0.31112246957066836, 0.030628546408488694, -0.18621008464171002, 0.0006580394384749631, 0.16368646985159763, -0.05189112180786898, 0.07362776476922749, 0.22072770183418386, -0.023628595340400772, 0.32239966348428756, 0.07702998187673363, 0.3586883446696578, -0.012658618208847623, 0.24854081686047172, 0.11105574599214994, 0.07270000391637925, 0.005429496801784928, 0.28038223255190825, 0.1292681838255846, 0.06067334032532329, 0.212879526827466, 0.1908979497489721, 0.00714974595804813, -0.04956967046090328, -0.003033708064135896, -0.14078142536114208, -0.15988407289896017, -0.07846241775889441, 0.2609379446534831, 0.047875023248627455, 0.04737874209315835, -0.06342895916322683, -0.1092484402367306, -0.17324098602615245, -0.17832613464472316, -0.20662880654146576, -0.17598374076350334, -0.16692023025916541, -0.11952374122543896, -0.33856292421599865, -0.4015784985365196, -0.2545679242961004, 0.1420617584181686, 0.015434343177825596, 0.012702477951740775, 0.288438311884168, 0.18984184228079323, -0.004681157741324566, -0.32128372628545665, 0.07169695106947492, 0.03731865472223733, 0.1640190102753582, 0.19545352478370506, -0.016045221882758302, 0.14294157259692566, -0.13531110018576956, -0.05108069557295929, 0.04595161710174542, 0.15253175885696066, -0.1658526045844391, 0.08984894395544187, 0.09794318730859268, 0.02489160061068668, -0.1442544466049594, -0.20034680375229083, 0.16908213274039896, -0.20280511061610207, 0.053716412893824915, 0.02334971235583445, -0.25377334873750784, 0.1474776738366479, -0.0156310474414721, -0.15009662809185903, 0.1348302168130506, -0.07898360326497685, 0.044165162702898844, -0.0650806454677019, 0.039742028003128926, -0.12993045538043357, 0.098004852922963, 0.13589812870068327, 0.12673607421460678, 0.0637490958658753, 0.034145763383090125, -0.054975419452954775, -0.01877146480129513, -0.32630823339422277, -0.09854518902544784, 0.057299297684057414, 0.0873699151773827, -0.040170802270212976, 0.15641552167927184, 0.06288721441229363, 0.0044285729568858, 0.07820700335237253, 0.12813364562003934, -0.05845890426913106, 0.48412554562148136, 0.004987933310264739, -0.05767382204908398, 0.02016640089046552, -0.1459692298282602, -0.25335801054156454, 0.2077489154817773, 0.08382681630450482, 0.06757319710278574, -0.0673538719552491, 0.055343004701133354, -0.11978545103596344, -0.02901555609801547, 0.07940912209123877, 0.17692275537395993, 0.12244396006856517, -0.19017275744951434, 0.15625960590962504, -0.24551990174051241, 0.4181928916874133, -0.10829542453236878, 0.08980856394070315, 0.08325212232441892, 0.09432208345694451, -0.019401180171606892, -0.19923991340632105, -0.20701844642077583, 0.24389851276706936, -0.11310006346270676, 0.10757899082242084, -0.0975437901374637, 0.11814009059919185, 0.2633903966258602, -0.030330538134425954, 0.09313039847305947, -0.18391408823122513, -0.33252235214441483, 0.03503963655246957, -0.1484057241236241, -0.03149600542816421, 0.18974865573167032, -0.0713574746334132, -0.24785104397278787, -0.13346444555258805, 0.1402381274710456, -0.19330619666715357, 0.2614569772077893, 0.013024230496752113, -0.0019281510650149022, -0.14773028476992534, -0.07572862709821856, -0.16563087467771712, 0.06834088989661989, 0.18540924606353795, 0.0635061501480098, 0.13463242418716825, -0.048185617701187004, 0.1563215374542997, 0.26435982495215865, -0.5048261381050358, 0.17636351134480244, -0.21039483678067597, 0.0032034477229185756, -0.030200286188629816, 0.23429239248439446, 0.3661357842649514, -0.2197839346303251, 0.38470979326324484, -0.026321920564355037, -0.10751351707469041, 0.1767470895556259, -0.13315666084422548, 0.012960144910400146, -0.018168709838488024, -0.029289912906178536, 0.3282207713503849, 0.16494859383310936, -0.06953538461502574, -0.03220254466528382, 0.10364206358071283, -0.06015514459699113, -0.22525562461144413, -0.08347409815187022, 0.0685890509820524, 0.09611116629619236, -0.3697796249678662, 0.494042074233782, -0.0064902411363504265, 0.10987235418281759, -0.09538788192244407, -0.10277829062998434, -0.00285318608002313, -0.08432870806190784, -0.34477230505081835, -0.17864178715003612, 0.08434833257747724, 0.02852484769146466, 0.04484165825192313, 0.19234109516359268, -0.004653837952926742, -0.10885977019890322, 0.2555828025606358, -0.29512980597721006, 0.24403868176373059, -0.0058928387561029945, 0.10405220287336145, 0.2756095133582074, 0.6913756809597406, 0.20792078633770936, 0.16356084010182573, 0.17548805773062853, 0.042102770759547256, 0.1449845566865475, 0.1582977404759918, -0.06196366581595166, 0.10093183049245036, -0.2653568225489072, -0.2833873118699403, -0.07820728696506625, -0.1749630645638106, -0.18890000663183884, -0.04148013790888919, 0.011089011504523964, 0.11020734642478398, -0.262244335811351, -0.07240778049963668, 0.05817642938410371, -0.10895767922877823, 0.4935214981367842, 0.08209991252487193, -0.23920005616596354, -0.20610209794694345, -0.1676227478573417, 0.12419254816463228, 0.30027048124510725, 0.4260303641200058, 0.118622121259245, -0.2421861402187244, -0.003559816934634133, 0.1475591844816296, -0.06029994357595618, 0.2172786656348842, 0.14814186774252916, 0.02991479193631311, 0.21059567268406307, -0.2151819962828112, 0.1058800840682492, -0.08394683546769165, 0.24403201283825934, 0.38968895892024025, 0.2056284915649741, 0.25555013856876946, -0.047257301872980904, -0.1631307586938957, -0.17515805846822355, -0.5134259365061673, -0.1592871462512714, 0.07445863011970388, 0.14289421016560333, 0.04834654497545201, -0.11767611120125179, 0.09986341978863908, 0.18067854503125358, -0.011878941113178831, 0.09367535877072806, 0.01341497295453433, -0.027668594946652653, 0.13298730493294442, -0.479498203838799, -0.24809143643416343, 0.20669676783106214, 0.2900007606109739, 0.18207268825829745, 0.13333420469964305, -0.1659681021769651, -0.22610883403851553, -0.5034793211378497, -0.21715569587994984, 0.05419093343900493, 0.1057725230801412, 0.10239781069102066, -0.055419829417097624, -0.17442489617570917, -0.07440661497842695, 0.1158883106799961, 0.010280436716236118, 0.4019335083227684, 0.039701061874317274, 0.25209381690678045, 0.012299462038946104, -0.16248943613013364, 0.1276429946986342, 0.06632545250840109, -0.09640036926570895, 0.13623814127568665, -0.007350201111405052, 0.179240018864051, 0.1759218528289525, 0.029232256689669046, -0.2959112723826984, -0.11403227399981508, 0.2064900600791224, 0.14142002118557054, -0.09799485054552994, -0.007929008908345147, 0.03298648706301872, -0.1216916561093957, -0.018846283720271854, 0.1324989986967636, 0.08375110643834141, 0.5368325664153107, 0.15579497579360022, -0.06044396260927856, -0.2102207412494946, -0.01747560409569011, -0.2763173410800081, -0.04720389077599416, -0.06626247551548654, 0.2770106418951775, 0.04414715123120932, -0.39797903647111293, -0.02801500232802413, 0.10068629145743752, 0.07024875739496167, 0.08250487412320008, 0.1647224690341783, -0.35710871273459915, -0.128563643774523, -0.14751931503061963, 0.5048978695422519, 0.08570540658606902, 0.5203021811997559, 0.24802255353765604, 0.04068814222705824, -0.0780726375303165, 0.010261224088853467, 0.015396380550148503, 0.15120791110596166, 0.23959920355191647, 0.017442782407454987, 0.09315838516990704, 0.24951984307753092, 0.2214806942301371, -0.04231635009822982, -0.1355334282594056, 0.05209760888078947, 0.04003889885124166, 0.018543932470516913, -0.32714892035264076, -0.4450972821528822, 0.1566200542943626, -0.21473068972393633, -0.5735290449816771, -0.0273831725109637, -0.236717595736536, -0.013216009790723173, -0.09088503694719721, 0.11701689611545008, -0.03536025027629386, -0.14443717781136606, 0.19876414128408854, -0.17048698631181167, 0.16445752566432773, 0.15707441777827796, 0.07514435674270334, 0.051111069227019974, -0.2097366397141196, -0.2595998617292569, 0.40471688782878623, -0.10360229563956076, 0.19826776267664092, -0.1540949328191183, -0.035860061225203854, -0.37099680685171016, -0.2075105548238832, -0.3956074482990594, -0.13293842914221768, -0.17630945551163962, 0.1958616133344684, 0.053023659295502054, -0.04681567247860786, -0.06622394373374559, -0.12147058929887354, 0.05341740870368301, -0.3864976416675618, 0.05189475154468671, -0.0041795275628335096, 0.05130032673446694, -0.023944387231397687, 0.1029647075334575, 0.2821680174223464, 0.10899368972592921, -0.11193049206246886, -0.07787928502968079, 0.2863842162115822, 0.0776036731930083, -0.5781236390929129, -0.10095420222895003, 0.005008663761134009, -0.10455913715546773, 0.2507331412966781, 0.0736637189623763, 0.046665449687497196, -0.3071961693728606, -0.02556278031064446, -0.025419310608050583, 0.08726401357818284, 0.10959030999919347, 0.15955462535391365, -0.08967520529641738, 0.01075131103039383, 0.42876639037634556, 0.16959503573882012, 0.008580564758532703, -0.028596760335521026, 0.6215131992386291, -0.19460013688426822, -0.047656309599784064, 0.21073210106676932, 0.3201087730333449, -0.2532073847233577, 0.34837933553290895, 0.46839617091634883, 0.015542415941611409, -0.13268511135889394, 0.04754066366560725, 0.34227834218729913, -0.06611442276593663, 0.07892368870394341, 0.03294419355996594, 0.1312535633812621, -0.3075695695025191, 0.08307204803571688, -0.18916973384220762, -0.13911074305028928, -0.31769859807944023, -0.14355679440855956, 0.11154556351381927, 0.12670262395997312, 0.1384676167711489, 0.43661540718084774, -0.3774284000525473, -0.13080138812272135, 0.1075602926419048, 0.29154229877875143, 0.3342245866368754, -0.24087111021043378, -0.09082730499469378, 0.09843346793467117, 0.31309974181621814, -0.36244508084631105, 0.2533326600462193, 0.010368485632181245, 0.06996868952595689, -0.19590725462321446, 0.008370803163527334, -0.12597148416755122, -0.3228869074374016, 0.08081256321010159, 0.37106348857790117, -0.013793874157468274, 0.044037237295168154, -0.2375325009975453, -0.4581413865534533, -0.5314323316742626, -0.3312342563344752, -0.30863690275283306, -0.20148929434738544, -0.33372161378276155, -0.0653041874793182, 0.021541009745754527, -0.22720788227293098, 0.2769026508309034, 0.2592969696565915, -0.11453031764300246, -0.04382167624305859, 0.27157489927884043, 0.10108369327561677, -0.1996406039843828, 0.026161565258125952, 0.9491608652047906, -0.03869334663868179, -0.2534231357154758, 0.2725748020599422, 0.04815993746560896, -0.355112702398512, -0.4476278085246375, -0.12592048794122182, 0.6170544581115214, 0.2506300786298572, -0.04403387133661372, -0.06932897237513823, -0.23449201728442665, -0.22930843123333824, 0.11218337077958283, -0.3768404160452991, 0.3359625467798377, 0.18505474533578822, 0.18991922156815286, 0.09680221436957809, -0.3723896721051202, 1.377688041508106, -0.23405358624405845, 0.012808534007506575, 0.27167043018568504, 0.3039217138592576, -0.0901911771342833, -0.25897848925249256, 0.19385352842175788, -0.05710278392138018, 0.3147462701715033, 0.06966006985721336, -0.17439951196965492, 0.13952926446808495, 0.20171030749273355, 0.12169951108896596, -0.5576175171757268, 0.2890396200360974, -0.13381626368690466, 0.03345961296151302, 0.26105694774705795, 0.29354931591080663, 1.4181334623967217, -0.35207535708292437, -0.035336430517620865, -0.0862226699510308, -0.4986415883095282, -0.1064190663512058, 0.051585518140460494, 0.08845550815033414, -0.23839518161657483, -0.11209495381863581, 0.42120690597261395, 0.09297912219477958, 0.1195222926031152, 0.11189807752085504, -0.37551793879021, -0.10261507419434104, -0.5977669329914541, -0.16804972769475965, 0.43369423648363, -0.13178177233797728, 0.2234983238187121, 1.320680811119729, 0.39777024447076137, 0.40002455154214955, 0.03307696883366677, -0.6739882810070296, -0.0032899054633633803, 0.1691865579464098, -0.31512870315234054, -0.057566167642493636, -0.23652469972984513, 0.22527023579334188, 0.07223454578340918, -0.24499748699516302, -0.19878497804071976, 0.11835363116374595, 0.15469505047094553, 0.01906285236726876, -0.2568395484208118, 0.6461606620942498, 0.07685142222711684, -0.026325052725483977, 0.8587844688857862, 1.0579262520188508, 0.29023696629357315, -0.04374983329192515, -0.14797934395759843, 0.7777240314524378, -0.11074616151322493, -0.49551151076570915, 0.004581510846095821, -0.22514155734882807, -0.182702635415895, -0.06435863594468684, 0.16324349774932517, 0.005194426106473754, 0.25643418172876087, 0.273993132792809, 0.3114456100633371, -0.2966493230824622, 0.3056672699233037, 0.04692680923555841, 0.07225158201392308, -0.7377238183217878, 0.8550337358383687, -0.37879967980015006, -0.1775624423529682, 0.3400924810624532, 0.7899508134643703, 0.20703090224388396, -0.031029010748400337, 0.028745971575633977, -0.20378888982629031, -0.3075288992980178, -0.08179731459597614, 0.16192119686167752, -0.36212781181078857, 0.2000168330241919, 0.5324463194443503, 0.319726447502745, 0.1637890046619929, -0.1851104765041544, 0.4796779548310034, -0.0981237163778997, -2.670625349333936, -0.14914523778709118, -0.13811755917581978, 0.11024350659165191, 0.8388962613488049, 0.13913068527249947, 0.7490800694640772, 0.9870345945328656, -0.023120742881382696, 0.3603033607121964, 0.09773003028413094, 0.36549713898704717, 0.3463291655268904, -0.29715129783163, -0.09691334222388343, 0.46710659936792803, 0.11443156913821584, 0.8121836793051108, -0.69517226321055, 0.15916617152401685, -0.22779041125627097, -3.640552812481259, -1.2373736029410751, -0.11566595258874537, 0.22600145101370683, 1.0977194999137831, -0.2752496693780767, -0.41604848191544297, 0.5868878551636713, -0.09013961213558302, 0.755813195047331, -0.4399624871681146, -0.6902902534031976, 0.19529826969488995, 0.18229681252633542, -1.1101034788657627, -1.277687769635078, -1.0800780853578331, 0.22525601570486792, -2.393338024339869, -1.8782977877855787, -0.029993973010557128, -3.642232233875038, -2.8943039412735843, -0.14623828656831392, -0.2567769566615573, -1.485364141664942, -2.747138713112059, -0.3931094800830678, -1.3906543745538724, 0.12284021892970182, -0.6494588571755372, 0.7852058722125164, 0.5863950789863923, -0.8670115646286005, 0.5448638543932188, 1.0593909544283642, 0.34787991296988413, 0.5534105595117641, -0.6216967064185793, 2.3996518398229814, 1.4179196482259808, 0.1888554504230753, 8.840281896588667, 3.056860627054894, 0.5414790728736386, -0.14737018058203588, -0.03545997272255803, 2.033220101840704]
common_intercept = -0.0011314764226027373
M_common = 65

common_intercept1 = -0.0035670228477590413
common_coef1 = [1.2571575278505256, -1.9322238861577798, 1.2351240123833869, -2.3395153947265093, 1.355123438922019, 1.1709744345947317, -0.3787976483079747, 1.2207309570751546, -1.7668754690386719, 0.16062708229768446, -1.1264264123855843, 0.38948275307617064, 1.5937447788842865, 0.8695771098284922, -1.966084282941292, 1.5605368758521672, -4.772377341608284, 5.841476905890357, -1.848593678623143, -2.0362814352462086, 2.396565498618738, -1.7190135734502225, 2.5615488953501604, -1.6495975663941498, -0.4390220638454143, 0.48961027644426597, 1.2240451362934697, -1.0675850618432523, 0.03392946820548473, -1.029174977125008, 2.036862909411884, -3.3729080236073186, 3.244463669651694, -1.3092328544012495, -0.816838613274528, 1.2899142185428656, 1.0025569149526499, -0.9060790072466315, -0.37491327480813935, -0.6135214935888884, 1.0968038978982597, 0.2846692104484729, -1.4419968651341675, 0.772383418313982, -1.849484439491888, 1.252094431614536, 0.2964653033146513, -0.34144600032919564, 0.5954313985257692, -1.820641211502116, 2.5236573492625123, -1.5605449656453214, 0.3003994614034793, 1.088247311864856, -1.7420013392705924, -0.8909863816385222, 0.6031018391355747, -0.7372353093962893, 1.6563853662770822, -2.6007797754645394, 2.3889734644793412, -0.4588036529877442, 1.4021273319577923, -1.8967449362340887, -1.7514380278763626, -2.0618689853515466, 3.8266258597651124, -1.221651306932666, 3.442571890370524, -1.6777723924908858, -3.5488364306269418, 2.8596776082381448, -8.060196896776711, 12.807436503919611, -8.107927388972492, -0.48279130767700656, 4.2762533355099, -3.587204529860681, 2.841974021835893, -4.908111168915888, 3.556245196592329, -1.644845701492876, 1.3517097817590964, -0.6568927626664604, -0.6624917576987982, -0.29540955235531186, 0.026044661519495475, 0.38046204503272585, -0.1529982848697128, -0.4623807215364328, -0.895705384883052, 1.6449242399752677, 0.5015198022347135, -2.6948564407524698, 0.43968566053140296, -0.0967286560656938, 0.521454750944552, 0.21834947642130517, -0.8551359737608716, 2.297589525879477, -3.238182366755139, 2.529351142315371, -0.2977074348338845, -0.6002032201016696, -0.3333144617928264, -0.48926888588386286, 0.6323235802498159, -1.6383467957923168, 0.3945645201817653, 2.684251496606351, -3.5089242955189857, 0.6496097538225633, 1.9529925735526354, 1.2439131501013585, -0.4101259971096584, -2.0473419383961087, 2.583716467970126, -2.2236167424073225, 0.33689459254768145, -2.2907146088211814, -0.9352005232768053, 0.22381629947900739, 5.614699561693607, -0.6113500613916115, 0.9931163466492063, -3.74100360363029, -1.2869219073044966, 0.6789083678204229, -3.8750408404263608, 15.559904363361191, -14.56867331726653, 3.4254739103718106, 2.9154492331277924, -4.176700939021003, 2.946508105743436, -4.644610691245413, 3.5832694671897563, -0.4898180927680602, 1.332712998110473, -1.7364069208572124, -0.6514982340083456, 1.0287693130276137, 0.5942448076808463, 0.41047870454284374, -2.3653104233219406, 1.4693148832036964, -2.893920524807413, 4.189795587881708, -3.641960346741806, 0.7976546111123982, -1.6139755184714426, 1.1027736368434886, 0.8809075090592119, -0.3554653185695082, -0.6741632897808443, 1.8886769694399461, -3.9031970457384344, 3.023248209792348, -1.416346125593359, -1.1983409081294274, 1.2398955489129455, -1.8346051762630955, 2.201152456038788, -2.8449702939335593, 2.2194812434598585, 0.07921252973891398, -0.621967472583172, -3.3451303478935035, 5.476467514067586, -2.4912339545089175, -0.8220582134294977, -1.4581794771542145, 1.7581052791145142, -1.2204643868963747, -0.6715469941608881, 2.6223357891911854, -0.7899265563687262, 1.1176010236525808, 2.6969390461341303, -1.5701964435376852, -3.6846755457087763, -1.8792916814090392, -1.4084151618385186, 1.5505053704277414, 0.09331803085936748, 18.89673961410181, -24.32797476328768, 9.413421753210423, 0.8345427718744436, -0.6192435467196997, -1.3559279133832571, 0.6472877300354489, 1.8809158423071985, -0.3110437443194228, -2.004358848985388, 1.0578675689122143, -0.8776180898229246, 1.2820319563697355, -0.34422630730405795, 0.436878086790401, 0.7892337868495132, -0.4268895022043076, 1.3063440515238318, 1.4564501941620613, -1.778308766710139, -0.7850215154374355, 2.032480272300078, 0.7981118048271715, -1.5435123649647184, -0.6229830167477254, -0.3119520953113692, 2.1149662690503246, -3.413210386014602, 2.938291338622629, 0.18300908195978627, 0.17402620249628592, 0.6253189976075204, 0.5058940118042903, 1.0087198431361717, -2.431739799016901, 2.5265030933446613, -1.5893150654992734, -0.7815816714949263, -2.049530319143135, 3.901909905732619, -3.087128130007561, 1.4032740119995126, -1.7357186840936214, 4.244623230757418, -4.0785454208927, 3.4227892869160654, 0.08533092940024055, 0.8788496864279004, -2.315680475504656, 0.9559316299262112, -2.8664170948587775, -2.6049781510369225, 0.2169042376570193, 1.5587216732124565, 4.596605257992853, -1.5870314271067218, 17.502230227953635, -35.22996110618268, 15.267959437783965, 1.0884049352072058, -0.9015504861289677, 1.5804281616555669, -2.3234334869079865, 3.602590055406509, -2.5033334373533878, 0.40622405461079575, 0.277895896546411, -0.38102480374059744, 0.47536951362453467, -0.8817502585637172, 0.7148780789610432, -1.4192876941332588, 0.7048948308361399, 0.3787709358608106, -1.0299988403470683, 0.9769485582876128, -0.6285546458424022, 3.739249064289224, -2.7776178351274825, 1.006278770297023, 0.2620643521189171, 0.025570549843679072, 1.6021447146411676, -2.3300920111659273, 1.848351347804467, -1.397082645378246, 1.581490353976909, -0.33555199900096405, -0.15759603153268553, 0.6820003234230801, -1.072620536556913, 2.821316234477783, -0.7921431995590723, 2.1627456048346785, -1.555178418551931, 4.476669904890402, -3.7658306828410306, 3.4331865900982446, -3.62568482126948, 4.28415095734588, -3.933493782907761, 2.152302707301575, -1.0962073114832624, 2.120716200185265, -1.2820893842282635, 0.39380811880651173, 1.4362683307121042, 0.34287116193658507, 3.3828757073251827, -1.0482912282484018, 7.603300768290733, -7.116470066444936, 11.621645118900787, -39.24907582733488, 29.296847914394235]
M_common1 = 300



class DelayedRLSPredictor:
	def __init__(self, n_channels, M=3, lambda_=0.999, delta=100, delay=0, mu=0.3):
		self._M = M
		self._lambda = lambda_
		self._delay = delay
		self._mu = mu
		size = M * n_channels
		self._w = np.zeros((size,))
		self._P = delta * np.eye(size)
		
		
		self.M2 = 60
		size2 = self.M2
		self._w2  = np.zeros((size2,))
		self._P2 = delta * np.eye(size2)
		self.regressors = deque(maxlen = self.M2 + delay + 1)
		#self.regressors = deque(maxlen = 3000)
		#self.regressors = []
	
	def append(self, sample):
		self.regressors.append(sample)
	def update(self):
		regressors = np.array(self.regressors)
		if regressors.shape[0] > self._delay + self._M:
			# predicted var x(t) 
			# это ПОСЛЕДНЯЯ строка, которую я получил. 
			predicted = regressors[-1, TARGET_CHANNEL]

			# predictor var [x(t - M), x(t - M + 1), ..., x(t - delay)]
			# читаю M последних строк (с учётом лага) в один вектор длины M * n_channels
			predictor = regressors[- self._M - self._delay - 1: - self._delay - 1].flatten()  #

			# update helpers
			pi = np.dot(predictor, self._P) # это вектор-строка - произведение вектора и матрицы XP
			# k = XP/(l + XPy) # нормируем полученное произведение
			k = pi / (self._lambda + np.dot(pi, predictor)) 
			# обновляем матрицу, вычтя матричное произведение векторов k*pi P = 1/l * (P -k*pi)
			self._P = 1 / self._lambda * (self._P - np.dot(k[:, None], pi[None, :]))

			# update weights
			dw = (predicted - np.dot(self._w, predictor)) * k
			self._w = self._w + self._mu * dw
	def update2(self):
		regressors = np.array(self.regressors)
		if regressors.shape[0] > self._delay + self.M2:
			predicted = regressors[-1, TARGET_CHANNEL]
			predictor = regressors[- self.M2 - self._delay - 1: - self._delay - 1, TARGET_CHANNEL].flatten()
			pi = np.dot(predictor, self._P2) # это вектор-строка - произведение вектора и матрицы XP
			k = pi / (self._lambda + np.dot(pi, predictor)) 
			self._P2 = 1 / self._lambda * (self._P2 - np.dot(k[:, None], pi[None, :]))
			dw = (predicted - np.dot(self._w2, predictor)) * k
			self._w2 = self._w2 + self._mu * dw
	def predict_linear(self):
		regressors = np.array(self.regressors)
		if regressors.shape[0] > self._delay + self._M:
			# return prediction x(t + delay)
			return np.dot(self._w, regressors[- self._M:].flatten())
		return 0 
	def predict_linear2(self):
		regressors = np.array(self.regressors)
		if regressors.shape[0] > self._delay + self.M2:
			return np.dot(self._w2, regressors[- self.M2:,TARGET_CHANNEL].flatten())
		return 0


rls = DelayedRLSPredictor(n_channels=N_CHANNELS, M=32, lambda_=0.9996, delta=0.01, delay=SKIP_SIZE, mu=1)


data1 = deque(maxlen = M_common  + SKIP_SIZE + 1)
data2 = deque(maxlen = M_common1 + SKIP_SIZE + 1)

def common_predict():
	regressors = np.array(data1)
	if regressors.shape[0] > SKIP_SIZE + M_common:
		return np.dot(common_coef, regressors[- M_common:,:].flatten()) + common_intercept
	return 0 
def common_predict1():
	regressors = np.array(data2)
	if regressors.shape[0] > SKIP_SIZE + M_common1:
		return np.dot(common_coef1, regressors[- M_common1:].flatten()) + common_intercept1
	return 0 

# читаю первые 3000 строк, обучающие, и каждый раз что-то предсказываю (и забываю)
for i in range(OFFSET):
	cur_data = list(map(float, input().split()))
	rls.append(cur_data)
	data1.append(cur_data)
	data2.append(cur_data[TARGET_CHANNEL])
	rls.update()
	rls.update2()
	

pr0, pr1, pr2, pr3 = 0, 0, 0, 0
w1 = 0.1
w2 = 0.1
w3 = 0.4

pr0 = common_predict()
pr3 = common_predict1()

pr1 = rls.predict_linear()
pr2 = rls.predict_linear2()

prediction = pr0 * (1-w1-w2-w3) + pr1 * w1 + pr2 * w2 + pr3 * w3

# вывожу предсказание после первых 3000 строк
print(prediction)
sys.stdout.flush()



# читаю очередную тестовую строку и делаю предсказание на её основе. 
while True:
	cur_data = list(map(float, input().split()))
	
	data1.append(cur_data)
	data2.append(cur_data[TARGET_CHANNEL])
	rls.append(cur_data)
	
	pr0 = common_predict()
	pr3 = common_predict1()
	pr1 = rls.predict_linear()
	pr2 = rls.predict_linear2()
	prediction = pr0 * (1-w1-w2-w3) + pr1 * w1 + pr2 * w2 + pr3 * w3
	print(prediction)
	sys.stdout.flush()


