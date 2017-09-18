if (scalar(@ARGV) ne 3)
{
  print STDERR "perl LiNMT-match-template-with-translation.pl TEMPLATE TRANSLATION RESULT\n";
  exit(1);
}

open (TEMPLATE, "<", $ARGV[0]) or die "Error: can not read $ARGV[0]!\n";
open (TRANSLATION, "<", $ARGV[1]) or die "Error: can not read $ARGV[1]!\n";
open (OUTPUT, ">", $ARGV[2]) or die "Error: can not write $ARGV[2]!\n";

my $line_num = 0;
my $total_temp_words = 0;
while (my $line_temp = <TEMPLATE>)
{
  ++$line_num;
  my $line_translation = <TRANSLATION>;

  $line_temp =~ s/[\r\n]//g;
  $line_translation =~ s/[\r\n]//g;

  my %hash_translation = ();
  
  my @translation_words = split /\s/, $line_translation;
  foreach my $word (@translation_words)
  {
    ++$hash_translation{$word};
  }
  my $words_number = keys %hash_translation;
  #print OUTPUT $line_num."\t".$words_number."\n";

  my @temp = split /\s/, $line_temp;
  my $true_word_number = 0;
  my $match_number = 0;
  foreach my $word (@temp)
  {
    if ($word eq "#")
    {
      next;
    }
    else
    {
      ++$true_word_number;
      if (exists $hash_translation{$word})
      {
        ++$match_number;
      }
    }
  }
  
  $total_temp_words += $true_word_number;
  $total_match += $match_number;
  print OUTPUT "LINE=$line_num\t1BEST=$words_number\tTEMP=$true_word_number\tMATCH=$match_number\n";


}

my $match_percent = $total_match / $total_temp_words;
print OUTPUT "TOTAL=$total_temp_words\tMATCH=$total_match\tP=$match_percent\n";

close (TEMPLATE);
close (TRANSLATION);
close (OUTPUT);
