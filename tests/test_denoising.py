from transformers import AutoTokenizer
from text_denoising.collate_fn import DataCollatorForUL2



if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("bigscience/mt0-small")
    tokenizer.add_special_tokens({'bos_token': '</s>'})

    # collate_fn = DataCollatorForUL2(tokenizer,
    #                             r_probability=1.0, r_denoising=True,
    #                             s_probability=0.0, s_denoising=False,
    #                             x_denoising=False, x_probability=0.0)
    collate_fn = DataCollatorForUL2(tokenizer,
                                r_probability=0.5, r_denoising=True,
                                s_probability=0.0, s_denoising=False,
                                x_denoising=True, x_probability=0.5)

    batch = [
        'Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nulla faucibus turpis et elit malesuada, ac venenatis sapien bibendum. Mauris at ullamcorper libero. Donec interdum auctor nisi a luctus. Suspendisse potenti. Proin vitae tortor vel leo consectetur fermentum. Sed blandit, nulla ac lobortis dapibus, diam massa accumsan velit, non pharetra lectus lacus ac odio. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia Curae; Sed viverra libero at est efficitur vestibulum. Sed ac tristique mauris, sit amet consequat leo. Sed imperdiet lectus a magna mollis auctor. Sed bibendum eget lacus vitae lobortis. Fusce ac eros eget libero scelerisque consequat. Cras id purus ornare, consectetur ipsum sed, semper nulla. Sed tincidunt eget neque eu pulvinar.. Sed tincidunt eget neque eu pulvinar.. Sed tincidunt eget neque eu pulvinar.. Sed tincidunt eget neque eu pulvinar.. Sed tincidunt eget neque eu pulvinar.. Sed tincidunt eget neque eu pulvinar.. Sed tincidunt eget neque eu pulvinar.',
        'Donec tincidunt enim quis felis lacinia, vitae ultricies nulla consectetur. Praesent ullamcorper ligula ac tincidunt rutrum. Vestibulum euismod ex vel quam porttitor, sit amet consequat velit sollicitudin. Pellentesque non mauris auctor, varius odio non, feugiat mi. Sed vulputate tincidunt arcu eget interdum. Duis convallis purus eget mauris euismod, non efficitur mi convallis. Sed varius massa nec luctus iaculis. Donec ornare, nunc a consequat pellentesque, nisi orci tincidunt quam, ac laoreet mauris orci in nunc. Fusce ut orci sit amet turpis vestibulum imperdiet. Vivamus et enim vel lorem ultrices fringilla. Sed vehicula nibh id risus convallis, ac bibendum sapien vulputate. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia Curae; Integer at risus quis magna blandit aliquam. Suspendisse varius malesuada mauris, vitae dictum metus aliquam vitae. Ut et ante at odio malesuada lobortis. . Sed tincidunt eget neque eu pulvinar.. Sed tincidunt eget neque eu pulvinar.',
        'Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed eleifend et tortor ac vehicula. Fusce eu urna aliquet, fringilla odio vel, malesuada metus. Pellentesque bibendum mauris vel est faucibus posuere. Duis ultrices vestibulum nulla, at tempor enim bibendum a. In sit amet quam vel nunc tristique varius ac eu dui. Quisque semper nisi at enim aliquam facilisis. Sed pharetra risus sit amet libero sollicitudin, vel faucibus velit sagittis. Sed viverra magna quis metus malesuada posuere. Donec in ante in enim tristique vestibulum. Sed molestie posuere urna id rhoncus. Fusce sit amet neque ac mi dapibus sollicitudin. Fusce pharetra est sed massa feugiat euismod. Vestibulum eu aliquam nulla, eget varius eros.. Vestibulum eu aliquam nulla, eget varius eros.. Vestibulum eu aliquam nulla, eget varius eros.. Vestibulum eu aliquam nulla, eget varius eros.. Sed tincidunt eget neque eu pulvinar.. Sed tincidunt eget neque eu pulvinar.. Sed tincidunt eget neque eu pulvinar.. Sed tincidunt eget neque eu pulvinar.. Sed tincidunt eget neque eu pulvinar.. Sed tincidunt eget neque eu pulvinar.',
        'Suspendisse malesuada nibh a enim blandit, a laoreet augue imperdiet. Suspendisse at ligula non risus feugiat blandit. Proin tristique mi sit amet ex laoreet, vel viverra ipsum fringilla. Donec at gravida nisi. Curabitur vel magna vitae lectus bibendum lacinia ac vel elit. Nam sit amet sem purus. Suspendisse at metus vitae ipsum viverra bibendum. Ut quis gravida libero. Suspendisse varius vel purus nec scelerisque. Nulla tincidunt enim in mollis eleifend. Donec tincidunt justo vitae diam congue vestibulum. Nam id lectus auctor, pellentesque tellus in, scelerisque tellus. Vivamus vel semper justo. Sed eget ipsum nec libero pellentesque tincidunt. Ut in efficitur purus, sit amet hendrerit ex.Ut in efficitur purus, sit amet hendrerit ex.Ut in efficitur purus, sit amet hendrerit ex.Ut in efficitur purus, sit amet hendrerit ex.Ut in efficitur purus, sit amet hendrerit ex.Ut in efficitur purus, sit amet hendrerit ex.. Sed tincidunt eget neque eu pulvinar.. Sed tincidunt eget neque eu pulvinar.',
    ]
    encode = collate_fn([ { 'input_ids': tokenizer(r)['input_ids'][:200] } for r in batch] )
    print(tokenizer.decode(tokenizer(batch[0])['input_ids']))
    print('-----')
    for input_ids, token_ids, label_ids in zip(encode['input_ids'], encode['decoder_input_ids'], encode['labels']):
        print('---------')
        print(tokenizer.decode(input_ids))
        print(tokenizer.decode(token_ids))
        print(tokenizer.decode(label_ids[label_ids!= -100]))
        print('---------')
