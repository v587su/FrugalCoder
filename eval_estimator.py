import torch
import pandas as pd
import lightning as L
import tqdm
import pickle
import os
import numpy as np
import random
random.seed(233)
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from train_estimator import load_neural_model

from sklearn import metrics
import time
from utils import arg_parser, get_latest_checkpoint, load_dataset, get_latest_lighting_checkpoint


PROPORTIONS = [0.05, 0.1, 0.2, 0.5, 0.8, 1.0]
# PROPORTIONS = [0.2, 1.0]
EVAL_IDS = {
    'python_codegen': [20588, 5713, 17208, 7563, 15863, 18048, 6696, 2728, 3352, 8803, 7664, 471, 2520, 22172, 7805, 19842, 7447, 9327, 3835, 15778, 10538, 7156, 2859, 9582, 15894, 17242, 20684, 4906, 19975, 10974, 10232, 3284, 16456, 14793, 8891, 13481, 1180, 13140, 5935, 1807, 2095, 3795, 6979, 5259, 11594, 10477, 16927, 17022, 14702, 4344, 861, 15256, 3859, 11365, 21668, 3293, 2657, 18534, 11823, 2036, 12834, 9795, 17771, 1297, 19962, 16484, 15144, 13193, 218, 2382, 15761, 11767, 12437, 4198, 20546, 14657, 11461, 9758, 19817, 4426, 6735, 16069, 17423, 22044, 4359, 13131, 4371, 6292, 19310, 16517, 3474, 1850, 10803, 13926, 15317, 6556, 20305, 10450, 19333, 11850, 4385, 22091, 9942, 16615, 19891, 7692, 2825, 17314, 3086, 1214, 5784, 20624, 18814, 13621, 19527, 8537, 7252, 3335, 9326, 1683, 17600, 12166, 10676, 13872, 21576, 6180, 12691, 21822, 1179, 12507, 13503, 20729, 6665, 9536, 8779, 3671, 19776, 20046, 15763, 18376, 9864, 12934, 13645, 13858, 8075, 3382, 3708, 7723, 21119, 3024, 12625, 16518, 19210, 12988, 6919, 19652, 13138, 12624, 17848, 14586, 17764, 5720, 887, 3570, 7877, 18644, 7824, 9991, 3831, 18473, 14029, 17102, 6407, 9802, 14705, 4303, 11173, 15661, 1659, 19774, 10982, 17234, 5704, 199, 17877, 13289, 8463, 18843, 1548, 11534, 6725, 19312, 14888, 2185, 20665, 14359, 17927, 19452, 18629, 11478, 1344, 8890, 11689, 19375, 21357, 18159, 17412, 11878, 9017, 20100, 12071, 14781, 22022, 1775, 22115, 8821, 19601, 4606, 19240, 11815, 14489, 8396, 1852, 6590, 3860, 2738, 20878, 14642, 3365, 5944, 3151, 2446, 19961, 17862, 20831, 16303, 4485, 1624, 8755, 2773, 19551, 9909, 10459, 8237, 4920, 8405, 12595, 13898, 1868, 149, 16039, 15862, 21754, 20841, 7687, 703, 13177, 10483, 20977, 18417, 20427, 21243, 20912, 12352, 20025, 9770, 20626, 4555, 7288, 20549, 21340, 20116, 1926, 2306, 5015, 884, 5196, 1698, 16077, 17990, 13082, 13651, 17900, 19258, 20357, 920, 19786, 7745, 2706, 21267, 11091, 12258, 8862, 1463, 9621, 13489, 17142, 19091, 9902, 2632, 20983, 6398, 7678, 3651, 3250, 7266, 18598, 11585, 11991, 9525, 12670, 3300, 16427, 2811, 9728, 20677, 15953, 256, 19522, 13630, 19078, 9069, 20659, 12450, 6758, 5327, 20702, 1671, 2287, 7933, 15119, 5508, 6304, 20738, 16734, 12765, 2873, 15739, 18803, 18698, 15174, 4895, 17908, 9734, 6894, 17469, 15169, 17151, 3437, 12199, 21666, 11339, 1493, 556, 1162, 22031, 9759, 3040, 833, 13039, 18374, 383, 14254, 16088, 9077, 8297, 14454, 15248, 6329, 14051, 9646, 20201, 12123, 14693, 19951, 193, 13498, 7234, 12114, 12531, 20186, 4942, 12298],
    'java_codegen': [63899, 24949, 63433, 17373, 568, 4780, 69861, 42553, 45559, 48239, 54617, 38523, 69212, 8365, 82258, 17139, 9662, 50485, 11342, 37336, 1535, 59022, 8204, 59074, 691, 84295, 18328, 37361, 27754, 70971, 18371, 64530, 35080, 44826, 76576, 15335, 37629, 61124, 14958, 84467, 77041, 17746, 873, 35426, 11784, 25932, 84074, 77414, 61515, 52472, 24318, 56868, 63325, 18363, 78838, 80485, 56277, 30023, 77605, 41487, 76251, 70585, 76078, 55540, 50645, 74197, 74590, 36939, 4218, 82431, 6421, 12384, 44618, 10108, 36210, 4194, 60796, 78681, 70263, 2045, 49165, 64191, 39895, 14240, 46650, 15618, 48132, 81598, 1286, 6354, 11828, 2363, 46276, 21539, 787, 4307, 21632, 21650, 17301, 8114, 46786, 57186, 49952, 55385, 71162, 65241, 67449, 7949, 66906, 1151, 74004, 70116, 48255, 18483, 34858, 77565, 59672, 11622, 56485, 77115, 24985, 53573, 52518, 57069, 45545, 53129, 55014, 19153, 64727, 3666, 57110, 40383, 18535, 59562, 40220, 69407, 16388, 40111, 35632, 9536, 28278, 76574, 72140, 1164, 63787, 59944, 77829, 4217, 62042, 80378, 22508, 15171, 57031, 30368, 77406, 83439, 81703, 79712, 9245, 51859, 19613, 62030, 35808, 3755, 74711, 5195, 18724, 14746, 23349, 54318, 30153, 1953, 55493, 13006, 64524, 80609, 22811, 72676, 79229, 73143, 20443, 72264, 27882, 56422, 32927, 41227, 81030, 50006, 13642, 55154, 61908, 37696, 67455, 81567, 22076, 65414, 34306, 44955, 82342, 74272, 80164, 211, 70515, 75403, 52475, 3100, 62358, 39451, 83339, 34197, 16173, 82394, 15990, 41980, 62694, 62460, 4952, 63018, 1120, 65132, 70029, 82502, 83146, 23719, 9691, 37190, 2604, 74237, 76155, 50059, 29946, 55888, 58952, 61724, 27446, 60514, 60185, 49193, 58339, 64965, 49724, 32409, 51493, 36170, 40370, 43316, 73941, 8300, 34110, 24237, 61424, 10950, 56565, 38583, 26463, 77013, 31753, 84537, 51539, 27470, 29409, 7189, 64084, 1981, 9761, 31206, 59345, 51932, 72320, 71599, 70648, 51771, 31581, 28922, 45911, 29570, 58580, 3729, 13342, 59724, 167, 56175, 54705, 66617, 73409, 78081, 76412, 55756, 40032, 21483, 31735, 72918, 20319, 31720, 10184, 13410, 23556, 78418, 4431, 25481, 71536, 34527, 13721, 43479, 84263, 41143, 29436, 15778, 65908, 16801, 53098, 62209, 13949, 43761, 30899, 51519, 58171, 43555, 78047, 80544, 11787, 76743, 47484, 57136, 64240, 79252, 12090, 53700, 33623, 3910, 76491, 73982, 27181, 71133, 39206, 76619, 84411, 43274, 58725, 19970, 46327, 13099, 55621, 35467, 7304, 80551, 17893, 77196, 58824, 75323, 1381, 22993, 49436, 20108, 75294, 43069, 41288, 15918, 76280, 41342, 75675, 3626, 37776, 27341, 51341, 26994, 73907, 22942, 46633, 35958, 17069, 41731, 39023, 53474, 64726, 6734, 63136, 68833, 45848, 18317, 47263, 8616, 58755],
    'python_starcoder': [4258, 3689, 8125, 3843, 14396, 4090, 6772, 5996, 12011, 1063, 3308, 9308, 8666, 17052, 20808, 11932, 4958, 18685, 14481, 10432, 13215, 3041, 2816, 12543, 2289, 10259, 7450, 841, 16558, 18568, 20589, 5088, 10990, 1030, 18054, 8028, 9845, 20547, 2147, 1487, 12796, 10028, 4962, 19363, 14273, 21106, 4059, 5073, 21335, 7010, 17937, 11469, 10090, 17125, 13872, 9563, 17705, 21323, 13507, 12687, 13460, 13284, 16009, 383, 10889, 21536, 20816, 12304, 14856, 153, 6201, 16520, 15143, 1454, 9078, 7019, 8935, 19459, 10314, 4634, 19283, 15614, 10920, 9783, 9952, 10558, 20561, 4111, 6497, 11081, 17260, 8722, 12646, 21607, 7188, 12465, 14346, 7017, 7288, 2275, 285, 15800, 574, 19613, 10356, 19273, 12054, 17356, 11604, 8178, 1755, 1809, 6392, 14680, 20753, 5116, 11094, 3842, 1798, 18389, 15148, 21075, 6788, 19467, 8335, 20564, 5727, 7330, 9511, 2466, 12395, 12914, 396, 8163, 1469, 20709, 15850, 9798, 16654, 21195, 18423, 13765, 187, 18177, 19322, 6535, 6023, 8308, 20867, 487, 6865, 6761, 19379, 8354, 20273, 18129, 21771, 10272, 12609, 3950, 2836, 19833, 7109, 14760, 645, 20900, 18672, 12713, 9534, 3456, 19207, 5670, 5669, 11821, 11167, 9580, 21921, 4710, 11221, 5152, 7878, 19140, 3467, 7640, 7121, 1634, 14874, 9108, 18314, 15947, 19636, 1936, 1986, 4190, 21160, 17160, 17381, 4677, 5585, 17133, 11727, 17479, 10726, 4296, 15773, 8550, 11612, 17291, 6686, 376, 15226, 18945, 10769, 875, 11968, 11317, 2301, 18995, 6026, 19182, 18000, 15820, 1252, 2818, 8472, 6729, 11392, 137, 17997, 4364, 20231, 3197, 12174, 4976, 6719, 8242, 17083, 21830, 16884, 3674, 18330, 15589, 3055, 12709, 13382, 20741, 9416, 17918, 8916, 16248, 21475, 8538, 2857, 10289, 13138, 18454, 3907, 2204, 9545, 2507, 7081, 8694, 20530, 9084, 10780, 2814, 18300, 1919, 9870, 8408, 21359, 17599, 14280, 19627, 6339, 5455, 12573, 19507, 7855, 16952, 6048, 12333, 11373, 3551, 3770, 390, 19641, 14447, 17198, 2177, 21545, 20885, 1201, 19044, 11765, 1336, 13125, 12671, 9442, 18336, 9420, 20090, 2536, 3169, 13591, 1231, 16643, 14976, 6290, 5685, 9431, 17873, 9100, 21485, 10363, 17638, 13481, 3238, 20270, 20247, 6868, 20078, 15828, 10758, 18233, 3424, 590, 17268, 1426, 13578, 14951, 8293, 9056, 17948, 18239, 21672, 11363, 21558, 7340, 8489, 2799, 2542, 12083, 14897, 16302, 6578, 1641, 7663, 3087, 18522, 9231, 9611, 1478, 3154, 21288, 8628, 7492, 8931, 1471, 6388, 13068, 406, 3008, 19764, 15145, 2208, 20504, 13057, 18846, 15676, 12301, 18132, 10410, 3518, 19206, 10163, 7094, 2614, 19191, 14277, 8943, 18578, 5350],
    'java_starcoder': [19415, 49682, 59111, 70098, 81912, 80441, 5067, 80101, 17147, 49664, 51536, 75819, 52209, 83076, 62539, 79369, 49871, 72723, 56250, 25515, 62928, 54823, 39565, 58246, 19365, 36022, 70587, 5384, 10942, 24637, 33470, 42840, 22518, 75420, 6727, 8298, 29770, 8572, 23223, 56200, 30746, 66154, 79238, 28896, 39023, 10348, 71578, 24754, 81247, 61622, 22043, 45065, 34193, 84905, 38173, 5750, 50685, 26731, 43797, 44775, 23217, 61011, 52078, 72590, 82897, 66778, 70914, 28417, 65722, 36496, 71265, 49537, 10186, 57531, 45495, 10689, 4353, 8818, 80154, 48003, 20001, 58879, 3800, 48002, 54371, 70010, 48240, 58269, 70529, 24951, 10561, 3545, 828, 49892, 81905, 76811, 66049, 22823, 17855, 27252, 39868, 41521, 62868, 84293, 74726, 40858, 56474, 21568, 55161, 71135, 1782, 56983, 2040, 30064, 62510, 59429, 490, 55295, 44689, 39797, 29665, 40114, 42438, 23386, 40596, 14118, 71921, 81776, 47094, 50203, 48837, 3932, 53859, 1263, 32806, 78921, 76319, 21356, 82295, 71107, 74422, 78947, 27611, 66528, 67593, 68703, 52688, 24599, 50584, 10579, 57991, 2218, 39729, 70960, 36139, 54376, 19326, 35933, 69967, 63902, 73646, 8226, 2408, 83914, 16447, 63964, 20858, 40590, 17045, 66047, 57460, 66461, 352, 35393, 10322, 81642, 64792, 33948, 48614, 72155, 52646, 8105, 68332, 34531, 21981, 58111, 2268, 19605, 68420, 70137, 28581, 66501, 47137, 54898, 54991, 45392, 75461, 32904, 82245, 71590, 59995, 67106, 42391, 12440, 7085, 79324, 62374, 83912, 80648, 75942, 43024, 66334, 57346, 7283, 60653, 30842, 53445, 31571, 83933, 32776, 29626, 75241, 68725, 21680, 32295, 60481, 37969, 30741, 12881, 3369, 64553, 17516, 3063, 66921, 60464, 7258, 1884, 54669, 25023, 16889, 23186, 18103, 47937, 42545, 19151, 82241, 61598, 43204, 8413, 50939, 63353, 1967, 2132, 81839, 4345, 53868, 54355, 7545, 19619, 28592, 52143, 67719, 2987, 6376, 62702, 8832, 9957, 44246, 52846, 82948, 46273, 71978, 35766, 52867, 77455, 47286, 81173, 21308, 73662, 72196, 35176, 37052, 81734, 31316, 82641, 70461, 25966, 62457, 56729, 78855, 30211, 22793, 1289, 77694, 64650, 52388, 66427, 35852, 66291, 4769, 83460, 68886, 82702, 83554, 51347, 13073, 33008, 54341, 9079, 35856, 61794, 42719, 53268, 2326, 15414, 68855, 79361, 59318, 3784, 42310, 48707, 7613, 80882, 30381, 73054, 69431, 658, 21292, 70548, 65646, 3359, 32052, 71110, 20765, 40927, 53526, 12187, 1291, 46441, 67741, 17544, 69592, 4190, 24443, 441, 39633, 50314, 36759, 62821, 37743, 1738, 49693, 41883, 8984, 36195, 502, 8239, 3526, 53182, 40330, 18673, 47392, 11882, 33810, 41152, 51740, 82557, 13133, 39393, 133, 13840, 66786, 12496, 64580, 10294, 13324, 83620, 16229, 45805, 68944, 74493, 19764, 22464]
}

def compute_errors(predictions, ground_truths):
    results = []
    # for i, thre in enumerate(thre_values):
    sorted_predictions_ids = np.argsort(predictions) # sort from small to large
    for p in PROPORTIONS:
        p_ids = sorted_predictions_ids[:int(len(predictions) * p)]
        
   
        mse = metrics.mean_squared_error(predictions[p_ids], ground_truths[p_ids])
        mae = metrics.mean_absolute_error(predictions[p_ids], ground_truths[p_ids])
        ground_truths_of_rejected = np.mean(ground_truths[p_ids])
        ground_truths_of_accepted = np.mean(ground_truths[sorted_predictions_ids[int(len(predictions) * p):]])
        results.append([mse, mae, ground_truths_of_rejected, ground_truths_of_accepted])
    return results

def compute_acc(predictions, ground_truths):
    # revert 0-1 to 1-0
    
    ground_truths = 1 - ground_truths
    results = []
    acc_predictions_ids = np.argsort(acc_predictions) # sort from small to large
    for p in PROPORTIONS:
        p_ids = acc_predictions_ids[:int(len(acc_predictions) * p)]
        binary_predictions = np.zeros(len(predictions))
        binary_predictions[p_ids] = 1
        # TP, FP, TN, FN
        
        tp = np.sum((binary_predictions==1) & (ground_truths==1))
        fp = np.sum((binary_predictions==1) & (ground_truths==0))
        tn = np.sum((binary_predictions==0) & (ground_truths==0))
        fn = np.sum((binary_predictions==0) & (ground_truths==1))
        print(f'TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}')
        random_predictions = np.random.randint(0, 2, len(predictions))
        print(f'Random TP: {np.sum((random_predictions==1) & (ground_truths==1))}, Random FP: {np.sum((random_predictions==1) & (ground_truths==0))}, Random TN: {np.sum((random_predictions==0) & (ground_truths==0))}, Random FN: {np.sum((random_predictions==0) & (ground_truths==1))}')
        false_positive_rate = fp / (fp + tn)
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        acc = (tp + tn) / (tp + tn + fp + fn)
        # accept rate of rejected samples
        filter_rejected = binary_predictions==1
        filter_accepted = binary_predictions==0
        
        rejected_ar = np.mean(1 - ground_truths[filter_rejected])
        accepted_ar = np.mean(1 - ground_truths[filter_accepted])
        original_ar = np.mean(1 - ground_truths)

        results.append([precision, recall, acc, rejected_ar, accepted_ar, original_ar])
    return results


def compute_num(predictions, ground_truths, precesions):
    # sort from small to large
    sorted_ids = np.argsort(predictions)
    
    # compute how many samples are needed to achieve the precision
    nums = []
    for p in precesions:
        current_num = 0
        current_precision = 0
        available_num = []
        for i in sorted_ids:
            current_num += 1
            if ground_truths[i] == 0:
                current_precision += 1
            if current_precision / current_num >= p:
                available_num.append(current_num)
        if len(available_num) == 0:
            nums.append(0)
        else:
            nums.append(max(available_num)/ len(predictions))
    return nums



if __name__ == '__main__':
    args = arg_parser()
    print(args)
    print(f'Estimator: {args.estimator}, Language: {args.language}, Model: {args.model}, Metric: {args.metric}')

    error_dataset, labels = load_dataset(args, 'test', need_score=True)
    labels = np.array(labels)

    file_name = f'{args.language}_{args.model}_{args.estimator}_{args.metric}'
    predictions = []
   
    if not args.load_prediction:
        if args.estimator in ['tcqe', 'bert', 'lstm']:
            checkpoint = os.path.join(args.output_dir, args.exp_name, file_name, 'best_model') if args.estimator in ['tcqe', 'bert'] else get_latest_lighting_checkpoint(os.path.join(args.output_dir, args.exp_name, file_name, 'lightning_logs', ))
            model, tokenizer = load_neural_model(args, checkpoint=checkpoint)
            def code_to_ids(batch):
                tokenized = tokenizer(batch['input_code'], padding=True, truncation=True, max_length=args.text_length)
                labels = batch['label']
                encoded = {'input_ids': tokenized['input_ids'], 'label': labels, 'attention_mask': tokenized['attention_mask']}
                return encoded
            error_dataset = error_dataset.map(code_to_ids, batched=True)
            model.eval()
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=32) if args.estimator in ['tcqe', 'bert'] else DataCollatorWithPadding(tokenizer=tokenizer, max_length=256, padding='max_length')
            
            data_loader = torch.utils.data.DataLoader(error_dataset.with_format(type='torch',columns=['input_ids', 'attention_mask']), batch_size=1, shuffle=False, collate_fn=data_collator)
            time_cost = 0
            for batch in tqdm.tqdm(data_loader):
                input_ids = batch['input_ids'].to(args.device)
                attention_mask = batch['attention_mask'].to(args.device)

                start_time = time.time()
                logits = model(input_ids, attention_mask=attention_mask).logits if args.estimator in ['tcqe', 'bert'] else model(input_ids)
                time_cost += time.time() - start_time
                score = logits.cpu().detach().numpy()[0][0]
                if score > 1:
                    score = 1
                elif score < 0:
                    score = 0
                predictions.append(score)
            print(f'Predict time: {time_cost}, Average time: {time_cost / len(data_loader)}')
            raise NotImplementedError
        elif args.estimator in ['lr', 'ada']:
            model = pickle.load(open(os.path.join(args.output_dir, args.exp_name, f'{file_name}.pickle'), 'rb'))
            error_dataset = error_dataset.with_format(type='numpy',columns=['input_code'])
            predictions = model.predict(error_dataset['input_code'])
            test_code = error_dataset['input_code'][0]
            start_time = time.time()
            _ = model.predict([test_code])
            print(f'Predict time: {time.time() - start_time}, Average time: {(time.time() - start_time) / len(error_dataset)}')
            raise NotImplementedError
        elif args.estimator in ['rand']:
            predictions = np.random.rand(len(error_dataset))
        else:
            raise NotImplementedError

        predictions = np.array(predictions)
        # labels = np.array(error_dataset['label'])
        # save predictions
        save_dir_path = os.path.join(args.data_dir, args.exp_name, 'test', 'evaluation')
        os.makedirs(save_dir_path, exist_ok=True)
        with open(os.path.join(save_dir_path, f'{args.language}_{args.model}_{args.estimator}_{args.metric}.pickle'), 'wb') as f:
            pickle.dump(predictions, f)
    else:
        try:
            with open(os.path.join(args.data_dir, args.exp_name, 'test', 'evaluation', f'{args.language}_{args.model}_{args.estimator}_{args.metric}.pickle'), 'rb') as f:
                predictions = pickle.load(f)
        except FileNotFoundError:
            with open(os.path.join(args.data_dir, args.exp_name, 'test', 'evaluation', f'{args.language}_{args.model}_{args.estimator}_{args.metric}.pkl'), 'rb') as f:
                predictions = pickle.load(f)

    acc_ids = EVAL_IDS[f'{args.language}_{args.model}']
    print(f'Acc ids: {acc_ids[:5]}')
    acc_predictions = predictions[acc_ids]
    annotation = pd.read_csv('./annotation_TOSEM.csv')
    acc_labels = annotation[f'{args.model}_{args.language}'].values
        
    acc_results = compute_acc(acc_predictions, acc_labels)
    
    accept_ids = acc_labels == 1 
    reject_ids = acc_labels == 0
    
    score_of_accept = predictions[acc_ids][accept_ids]
    score_of_reject = predictions[acc_ids][reject_ids]
    
    labels_of_accept = labels[acc_ids][accept_ids]
    labels_of_reject = labels[acc_ids][reject_ids]

    print(f'Accept Score: {np.mean(score_of_accept)}, Reject Score: {np.mean(score_of_reject)}')
    print(f'Accept Label: {np.mean(labels_of_accept)}, Reject Label: {np.mean(labels_of_reject)}')

    results = compute_errors(predictions, labels)


    if not os.path.exists('./eval_results.csv'):
        with open('./eval_results.csv', 'w') as f:
            f.write(f'language,model,estimator,metric,proportion,mse,mae,ground_truths_of_rejected,ground_truths_of_accepted,precision,recall,acc,mean_score_of_accept,mean_score_of_reject,mean_label_of_accept,mean_label_of_reject,rejected_ar,accepted_ar,original_ar\n')
        
    # save results
    with open('./eval_results.csv', 'a+') as f:
        for i, (r,ar) in enumerate(zip(results, acc_results)):
            mse, mae, ground_truths_of_rejected, ground_truths_of_accepted = r
            precision, recall, acc, rejected_ar, accepted_ar, original_ar = ar

            print(f'{args.language},{args.model},{args.estimator},{args.metric},{PROPORTIONS[i]}, {mse}, {mae}, {ground_truths_of_rejected}, {ground_truths_of_accepted}, {precision}, {recall}, {acc}, {np.mean(score_of_accept)}, {np.mean(score_of_reject)}, {np.mean(labels_of_accept)}, {np.mean(labels_of_reject)}, {rejected_ar}, {accepted_ar}, {original_ar}')
            f.write(f'{args.language},{args.model},{args.estimator},{args.metric},{PROPORTIONS[i]}, {mse}, {mae}, {ground_truths_of_rejected}, {ground_truths_of_accepted}, {precision}, {recall}, {acc}, {np.mean(score_of_accept)}, {np.mean(score_of_reject)}, {np.mean(labels_of_accept)}, {np.mean(labels_of_reject)}, {rejected_ar}, {accepted_ar}, {original_ar}\n')

            

        