exclude_p = dict(
	strip_space=True,
	irre_cls_list=[' ', ' z', 'ERROR', 'back', 'back ', 
					'cbtl', 'cbtl ', 'cgtl', 'cgtl ', 'crth',
					'crtl', 'cytl', 'cytl ', 'lanebtl', 'lanegtl',
					'lanertl', 'laneytl', 'lgtl', 'lrtl', 'lytl',
					'lytl ', 'otl', 'rgtl', 'rgtls', 'rytl',
					'sgtl', 'sgtls', 'srtl', 'sytl', 'tas',
					'trtl', 'tytl'],
	i_cls_include_list=['i1', 'i1 ', 'i10', 'i10 ', 'i11', 
						'i12', 'i13', 'i13 ', 'i14', 'i15', 
						'i2', 'i2 ', 'i3', 'i4', 'i4s', 
						'i5', 'i6', 'i9', 'il', 'il100', 
						'il3 0', 'il30', 'il40', 'il50', 'il60', 
						'il70', 'il80', 'il90', 'io', 'io ', 
						'ip', 'ip '],
	w_cls_include_list=['w1', 'w10', 'w11', 'w12', 'w13', 
						'w16', 'w16 ', 'w18', 'w19', 'w20', 
						'w21', 'w22', 'w22 ', 'w22r', 'w3', 
						'w3 ', 'w30', 'w32', 'w32 ', 'w32r', 
						'w32s', 'w34', 'w35', 'w38', 'w41', 
						'w42', 'w43', 'w45', 'w47', 'w55', 
						'w57', 'w57 ', 'w59', 'w63', 'w66', 
						'w8', 'w9', 'wo', 'wo '],
	panel_cls_list=['lo', 'lo ', 'ors', 'ors ', 'rn', 
					'rn ', 'ro'])

# p_finegrained = dict(
# 	ignore_number=True,
# 	sign_with_value=['pl', 'pm', 'ph'])

p_finegrained = dict(
	strip_space=True,
	ignore_number=True,
	# merge first then filter
	sign_merge_list=['pl', 'pm', 'ph', 'pa', 'pw'],
	# anns like 'p11`' with prime suffix are ignored 
	# better create a filter-out cfg to collect stats on the 
	# anns that are ignored
	p_cls_include_list=['p10', 'p11', 'p19', 'p20', 'p23',
						'p26', 'p5', 'pg', 'pn', 'pne', 
						'ps', 'ph', 'pl', 'pm', 'po'])

og_cls_list = [' ', ' ip', ' z', 'ERROR', 'back', 'back ', 'cbtl', 'cbtl ', 'cgtl', 'cgtl ', 
 			   'crth', 'crtl', 'cytl', 'cytl ', 'i1', 'i1 ', 'i10', 'i10 ', 'i11', 'i12', 
 			   'i13', 'i13 ', 'i14', 'i15', 'i2', 'i2 ', 'i3', 'i4', 'i4s', 'i5', 
 			   'i6', 'i9', 'il', 'il100', 'il3 0', 'il30', 'il40', 'il50', 'il60', 'il70', 
 			   'il80', 'il90', 'io', 'io ', 'ip', 'ip ', 'lanebtl', 'lanegtl', 'lanertl', 'laneytl', 
 			   'lgtl', 'lo', 'lo ', 'lrtl', 'lytl', 'lytl ', 'ors', 'ors ', 'otl', 
 			   'p0', 'p1', 'p1 ', 'p10', 'p10 ', 'p11', 'p11 ', 'p11`', 'p11s', 'p12', 'p12ss', 
 			   'p13', 'p13s', 'p14', 'p16', 'p17', 'p18', 'p19', 'p19 ', 'p2', 'p20', 
 			   'p20 ', 'p21', 'p22', 'p23', 'p23 ', 'p23s', 'p24', 'p24 ', 'p25', 'p26', 
 			   'p26 ', 'p26ss', 'p27', 'p28', 'p3', 'p3 ', 'p30', 'p4.5', 'p5', 'p5s', 'p6', 
 			   'p7', 'p8', 'p9', 'pa', 'pa10', 'pa12', 'pa13', 'pa13 ', 'pa13s', 'pa14', 'pa14S', 
 			   'pa20', 'pa30', 'pa36', 'pa5', 'pa55', 'pa6', 'pa7', 'panel', 'pao', 'pao10', 'pas', 
 			   'pb', 'pb ', 'pbtl', 'pg', 'pg ', 'pgtl', 'pgtl ', 'ph', 'ph ', 'ph 43', 'ph1.8', 
 			   'ph2', 'ph2.0', 'ph2.1', 'ph2.2', 'ph2.3', 'ph2.4', 'ph2.5', 'ph2.6', 'ph2.7', 'ph2.8', 
 			   'ph2.9', 'ph2.9 ', 'ph2o', 'ph3', 'ph3.0', 'ph3.1', 'ph3.15', 'ph3.2', 'ph3.3', 'ph3.4', 
 			   'ph3.5', 'ph3.6', 'ph3.7', 'ph3.8', 'ph3.9', 'ph3.95', 'ph3o', 'ph4', 'ph4.0', 'ph4.1', 
 			   'ph4.2', 'ph4.3', 'ph4.4', 'ph4.5', 'ph4.5 ', 'ph4.6', 'ph4.7', 'ph4.8', 'ph4.9', 'ph4o', 
 			   'ph5', 'ph5  ', 'ph5.0', 'ph5.1', 'ph5.2', 'ph5.3', 'ph5.5', 'ph5.5 ', 'ph5.7', 'ph5o', 
 			   'ph6', 'ph6.0', 'ph6.5', 'ph7.2', 'pho', 'pl', 'pl ', 'pl 30', 'pl 60', 'pl r', 'pl10', 
 			   'pl100', 'pl110', 'pl12', 'pl120', 'pl140', 'pl15', 'pl15 ', 'pl20', 'pl20 ', 'pl25', 
 			   'pl2o', 'pl3.5', 'pl30', 'pl30r', 'pl35', 'pl4', 'pl40', 'pl40 ', 'pl40r', 'pl49', 'pl5', 
 			   'pl50', 'pl50 ', 'pl55', 'pl60', 'pl60 ', 'pl70', 'pl70 ', 'pl70Es', 'pl80', 'pl80 ', 
 			   'pl90', 'plm10', 'pls', 'pm', 'pm ', 'pm10', 'pm13', 'pm15', 'pm20', 'pm25', 'pm26', 
 			   'pm3', 'pm30', 'pm36', 'pm40', 'pm49', 'pm49 ', 'pm5', 'pm50', 'pm55', 'pm7', 'pm70', 
 			   'pmo', 'pn', 'pn ', 'pne', 'pne ', 'pnl', 'pns', 'po', 'po ', 'pos', 'pot', 'pp', 'pr15', 
 			   'pr20', 'pr30', 'pr40', 'pr60', 'pr80', 'prtl', 'ps', 'ps ', 'pw', 'pw2', 'pw2.1', 'pw2.2', 
 			   'pw2.5', 'pw2o', 'pw3', 'pw3.5', 'pw4', 'pw6', 'pw7.8', 'pwo', 'pytl', 
 			   'rgtl', 'rgtls', 'rn', 'rn ', 'ro', 'rytl', 'sc', 'sgtl', 'sgtls', 'srtl', 'sytl', 'tas', 'tl', 'trtl', 
 			   'tytl', 'w1', 'w10', 'w11', 'w12', 'w13', 'w16', 'w16 ', 'w18', 'w19', 'w20', 'w21', 'w22', 
 			   'w22 ', 'w22r', 'w3', 'w3 ', 'w30', 'w32', 'w32 ', 'w32r', 'w32s', 'w34', 'w35', 'w38', 'w41', 
 			   'w42', 'w43', 'w45', 'w47', 'w55', 'w57', 'w57 ', 'w59', 'w63', 'w66', 'w8', 'w9', 'wo', 'wo ']
